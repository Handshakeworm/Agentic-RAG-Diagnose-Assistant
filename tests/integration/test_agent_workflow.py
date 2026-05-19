"""tests/integration/test_agent_workflow.py — F15 全工作流集成测试(DEV_SPEC §8.3 F15)。

Mock LLM(各节点的 get_llm)+ Mock 存储 / Embedding / Reranker。
覆盖两条无 interrupt 的可执行路径:
  1. 正常 confirmed:信息充足 → 直接诊断 → safety_gate → advice → format → END
  2. 追问触顶兜底:followup_round 预设到 MAX → diagnose Step -1 → failure_reason

涉及 interrupt 的追问 / 检查循环依赖 LangGraph Command(resume=...) 协议,
留给 F-J 端到端测;本文件聚焦"无 interrupt 直达终态"的 graph 编排正确性。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from src.agent.schemas.advice import AdviceOutput, MedicationAdvice
from src.agent.schemas.diagnosis import DiagnosisOutput, RankedDisease
from src.agent.schemas.info_collect import (
    InfoCollectOutput,
    PresentIllnessSlots as SchemaSlots,
)
from src.agent.schemas.ner import NEREntity, NERResult
from src.agent.schemas.query_construction import QueryConstructionOutput
from src.agent.schemas.safety_gate import SafetyGateOutput
from src.agent.state import create_initial_state


@pytest.fixture
def stub_dbs():
    """统一 patch 所有 DB / 模型调用,让 graph 跑在纯 mock 数据上。"""
    patches = [
        patch(
            "src.agent.nodes.info_collect.load_medical_history",
            return_value={"allergy_history": []},
        ),
        patch(
            "src.agent.nodes.info_collect.load_initial_exam_reports",
            return_value=[],
        ),
        # EL 移除后不再 patch terms_collection / embedding / build_sparse_queries —
        # build_query 无 alias 反查,sparse_queries 由 state 多字段直采
        patch(
            "src.agent.nodes.retrieve.search_dense_route",
            return_value=[
                {
                    "id": "d1",
                    "source_chunk_id": "c1",
                    "vector_type": "original",
                    "original_content": "胆囊炎诊断标准...",
                    "source_id": "src1",
                    "score": 0.9,
                }
            ],
        ),
        patch(
            "src.agent.nodes.retrieve.search_sparse_routes",
            return_value=[
                [
                    {
                        "id": "s1",
                        "source_chunk_id": "c1",
                        "vector_type": "original",
                        "original_content": "胆囊炎症状",
                        "source_id": "src1",
                        "score": 0.8,
                    }
                ]
            ],
        ),
        patch(
            "src.agent.nodes.retrieve.lookup_chunk_summary_question",
            return_value={},
        ),
        # 直接 stub graph 持有的 ④ ⑤ 函数引用 — 避免 TF-IDF 在 mock chunks 上抓出
        # 一堆碎片关键词、⑤ 逐个调 askability 把 mock 队列耗光。集成测的重点是
        # graph 编排(进出节点 + 路由判断),不是 ④/⑤ 内部逻辑(已在 unit 测覆盖)
        patch(
            "src.agent.graph.extract_symptoms",
            side_effect=lambda state: {"extracted_symptoms": []},
        ),
        patch(
            "src.agent.graph.select_discriminative_symptom",
            side_effect=lambda state: {
                "followup_questions": [],
                "unaskable_symptoms": [],
                "info_gain": 0.0,
            },
        ),
        patch(
            "src.agent.nodes.diagnose.rerank_with_fallback",
            return_value=[0],
        ),
        patch(
            "src.agent.nodes.diagnose.lookup_chunk_content",
            return_value={
                "c1": {
                    "chunk_raw_text": "胆囊炎诊断标准全文",
                    "parent_chunk_id": None,
                }
            },
        ),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def _patch_all_llms(factory_chain_obj, free_chain_obj):
    """把所有节点的 get_llm() 替换成同一个 chain 实例。"""
    factory = MagicMock()
    factory.return_value.with_structured_output.return_value.with_retry.return_value = (
        factory_chain_obj
    )
    factory.return_value.with_retry.return_value = free_chain_obj
    nodes = [
        "src.agent.nodes.info_collect.get_llm",
        "src.agent.nodes.build_query.get_llm",
        "src.agent.nodes.diagnose.get_llm",
        "src.agent.nodes.safety_gate.get_llm",
        "src.agent.nodes.generate_advice.get_llm",
        "src.agent.nodes.format_response.get_llm",
        "src.agent.nodes.select_symptom.get_llm",
        "src.agent.nodes.generate_followup.get_llm",
        "src.agent.nodes.process_followup.get_llm",
        "src.agent.nodes.recommend_exam.get_llm",
    ]
    return [patch(n, return_value=factory.return_value) for n in nodes]


# ────────────────────────────────────────────────────────────────────────────
# 路径 1:正常 confirmed → safety_gate → advice → format → END
# ────────────────────────────────────────────────────────────────────────────


def test_normal_confirmed_path(stub_dbs):
    from src.agent.graph import build_app

    schema_slots = SchemaSlots(
        onset_time="3天前", onset_mode="急性", trigger="进食",
        location="上腹", nature="胀痛", severity="中",
        duration_pattern="持续性", aggravating=["进食"], relieving=["热敷"],
        associated_symptoms=["反酸"], progression="加重",
        treatment_tried="奥美拉唑", treatment_response="部分缓解",
    )

    structured_invokes = [
        # ① info_collect (Step1 LLM)
        InfoCollectOutput(
            chief_complaint="腹痛 3 天",
            present_illness="进食后上腹胀痛",
            present_illness_slots=schema_slots,
        ),
        # ② NER
        NERResult(entities=[
            NEREntity(text="腹痛", entity_type="symptom", negation=False),
        ]),
        # ② EL: 三层归一化(无 LLM,Tier 1 走 query_term_by_alias_exact mock 命中)
        # ② Query
        QueryConstructionOutput(dense_query="进食后上腹胀痛"),
        # ⑤ select_symptom: slots 全填,无维度选择;extracted_symptoms 为空
        # (extract_symptoms 在没有 chunks_text 关键词时返回空) → 也无 askability
        # ⑩ diagnose 1 步 LLM 输出
        DiagnosisOutput(results=[
            RankedDisease(
                disease="胆囊炎", probability=0.85,
                evidence=["典型表现"],
                differentiation=None,
                differentiation_type="confirmed",
            )
        ]),
        # ⑪ safety_gate
        SafetyGateOutput(additional_risks=[]),
        # ⑫ advice
        AdviceOutput(
            medications=[
                MedicationAdvice(
                    drug_name="头孢克肟", dosage="100mg", frequency="每日2次",
                    duration="7天", notes="餐后服用",
                )
            ],
            risk_warnings=["饮食清淡"],
        ),
    ]

    advice_text_msg = MagicMock(); advice_text_msg.content = "您可能患有胆囊炎,建议..."
    free_text_invokes = [advice_text_msg]  # ⑬ format_response

    structured_chain = MagicMock()
    free_chain = MagicMock()

    sq = list(structured_invokes)
    fq = list(free_text_invokes)

    def s_invoke(prompt, config=None):
        if not sq:
            raise RuntimeError("structured chain queue exhausted")
        v = sq.pop(0)
        if isinstance(v, BaseException):
            raise v
        return v

    def f_invoke(prompt, config=None):
        if not fq:
            raise RuntimeError("free chain queue exhausted")
        return fq.pop(0)

    structured_chain.invoke.side_effect = s_invoke
    free_chain.invoke.side_effect = f_invoke

    patches = _patch_all_llms(structured_chain, free_chain)
    for p in patches:
        p.start()
    try:
        app = build_app()
        config = {"configurable": {"thread_id": "test-confirmed"}}
        state_in = create_initial_state(patient_id="P", patient_input="腹痛 3 天")
        result = app.invoke(state_in, config=config)
    finally:
        for p in patches:
            p.stop()

    assert result["diagnosis_result"][0]["disease"] == "胆囊炎"
    assert result["diagnosis_result"][0]["differentiation_type"] == "confirmed"
    assert result["diagnosis_result"][0]["failure_reason"] is None
    assert result["final_response"]


# ────────────────────────────────────────────────────────────────────────────
# 路径 2:追问触顶兜底
# ────────────────────────────────────────────────────────────────────────────


def test_followup_round_capped_path(stub_dbs):
    """followup_round 预设到上限 → diagnose Step -1 短路 → failure_reason='followup_round_capped'。"""
    from src.agent.graph import build_app

    schema_slots = SchemaSlots()  # 全部空槽

    structured_invokes = [
        # ① info_collect
        InfoCollectOutput(
            chief_complaint="腹痛",
            present_illness="x",
            present_illness_slots=schema_slots,
        ),
        # ② build_query:check path(followup_round == last_nlu_round 且非首轮)→
        #     跳 NER + EL,只跑 Step 4 Query 构建
        QueryConstructionOutput(dense_query="x"),
        # ⑤ select_symptom 已被 stub 替代(无 LLM 调用)
        # ⑩ Step -1 触顶,跳过 LLM(0 个)
        # ⑪ safety_gate
        SafetyGateOutput(additional_risks=[]),
        # ⑫ advice
        AdviceOutput(risk_warnings=[]),
    ]
    advice_text_msg = MagicMock(); advice_text_msg.content = "..."
    free_text_invokes = [advice_text_msg]

    structured_chain = MagicMock()
    free_chain = MagicMock()
    sq = list(structured_invokes)
    fq = list(free_text_invokes)

    def s_invoke(prompt, config=None):
        v = sq.pop(0)
        if isinstance(v, BaseException):
            raise v
        return v

    def f_invoke(prompt, config=None):
        return fq.pop(0)

    structured_chain.invoke.side_effect = s_invoke
    free_chain.invoke.side_effect = f_invoke

    patches = _patch_all_llms(structured_chain, free_chain)
    for p in patches:
        p.start()
    try:
        app = build_app()
        config = {"configurable": {"thread_id": "test-capped"}}
        state_in = create_initial_state(patient_id="P", patient_input="腹痛")
        state_in.followup_round = settings.agent_limits.MAX_FOLLOWUP_ROUNDS
        # check path:同时把 last_nlu_round 设到 MAX,build_query 走 Step-4-only 路径,
        # 避免 NER/EL 消耗多余的 mock 队列项
        state_in.last_nlu_round = settings.agent_limits.MAX_FOLLOWUP_ROUNDS
        result = app.invoke(state_in, config=config)
    finally:
        for p in patches:
            p.stop()

    assert result["diagnosis_result"][0]["failure_reason"] == "followup_round_capped"
    assert result["diagnosis_result"][0]["differentiation_type"] == "insufficient"
    assert any("本次问诊轮次较多" in r for r in result["risk_warnings"])
