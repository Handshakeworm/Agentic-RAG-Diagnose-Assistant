"""tests/unit/test_node_build_query.py — F3 ② build_query 单元测试(DEV_SPEC §4.1.2 ②)。

build_query 简化为三步(无 EL):
  Step 1 NER(LLM)→ symptom 实体按 negation 直接 raw text 进 confirmed/denied
  Step 2 Sparse 多字段直采(state 字段拼接,零 LLM,零 terms_collection)
  Step 3 Dense Query LLM 改写

只需 mock LLM(两处:NER / Query construction)。

覆盖:
- 首轮:三步全跑,chief/present NER 抽到的 symptom 进 confirmed/denied(raw text)
- 后续轮(非检查路径):对 followup_answer NER + last_nlu_round 推进
- 检查路径(followup_round == last_nlu_round 但非首轮):跳过 Step 1,只跑 Step 3
- confirmed_symptoms 去重(同一 raw text 不重复追加)
"""
from __future__ import annotations

from unittest.mock import patch

from src.agent.schemas.ner import NEREntity, NERResult
from src.agent.schemas.query_construction import QueryConstructionOutput
from src.agent.state import create_initial_state


def _setup_llm_mocks(mock_llm_factory, ner_entities, qc_dense):
    """组装两个 LLM 调用返回:NER → QueryConstruction(Step 2 多字段直采无 LLM)。"""
    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = [
        NERResult(entities=ner_entities),
        QueryConstructionOutput(dense_query=qc_dense),
    ]


@patch("src.agent.nodes.build_query.get_llm")
def test_first_round_full_pipeline(mock_llm_factory):
    """首轮:NER → Step2 多字段直采 → Step3;chief 中 symptom 实体按 negation 直接进 confirmed/denied(raw text)。"""
    from src.agent.nodes.build_query import build_query

    _setup_llm_mocks(
        mock_llm_factory,
        ner_entities=[
            NEREntity(text="肚子疼", entity_type="symptom", negation=False),
            NEREntity(text="发烧", entity_type="symptom", negation=True),
        ],
        qc_dense="持续3天的中等程度腹痛",
    )

    s = create_initial_state(patient_id="P1", patient_input="肚子疼3天没发烧")
    s.chief_complaint = "腹痛 3 天"
    s.present_illness = "肚子疼 3 天,没发烧"
    update = build_query(s)

    # 无 EL:NER raw text 直接进,不归一为 preferred_term
    assert "肚子疼" in update["confirmed_symptoms"]
    assert "发烧" in update["denied_symptoms"]
    assert update["dense_query"] == "持续3天的中等程度腹痛"
    # Step 2 多字段直采:chief_complaint 进 sparse(无 slots / report 时只有这一条)
    assert "腹痛 3 天" in update["sparse_queries"]
    assert update["last_nlu_round"] == 0  # followup_round 初始为 0
    # standardized_entities 字段已删,update 中不应出现
    assert "standardized_entities" not in update


@patch("src.agent.nodes.build_query.get_llm")
def test_check_path_skips_ner(mock_llm_factory):
    """检查路径(followup_round == last_nlu_round 且非首轮)只跑 Step 3。"""
    from src.agent.nodes.build_query import build_query

    # 只准备 1 个 invoke 返回 — Step 3 唯一 LLM 调用
    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = QueryConstructionOutput(
        dense_query="附加证据后的复合 query",
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_round = 2
    s.last_nlu_round = 2  # 检查路径标志
    s.chief_complaint = "腹痛"
    s.confirmed_symptoms = ["肚子疼"]  # 之前已建立的 raw text
    update = build_query(s)

    # 只调过 1 次 LLM(Step 3)
    assert mock_chain.invoke.call_count == 1
    assert update["dense_query"] == "附加证据后的复合 query"
    # last_nlu_round 不前进(NER 没跑)
    assert "last_nlu_round" not in update
    # confirmed_symptoms 透传(不动)
    assert update["confirmed_symptoms"] == ["肚子疼"]


@patch("src.agent.nodes.build_query.get_llm")
def test_dedup_appends_only_new_raw_text(mock_llm_factory):
    """首轮 NER 抽到与现有 confirmed_symptoms 重复的 raw text 不再追加。"""
    from src.agent.nodes.build_query import build_query

    _setup_llm_mocks(
        mock_llm_factory,
        ner_entities=[
            NEREntity(text="腹痛", entity_type="symptom", negation=False),
        ],
        qc_dense="x",
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.chief_complaint = "腹痛"
    s.confirmed_symptoms = ["腹痛"]  # 已存在
    update = build_query(s)
    # 没有重复追加
    assert update["confirmed_symptoms"] == ["腹痛"]


@patch("src.agent.nodes.build_query.get_llm")
def test_non_symptom_entities_not_promoted(mock_llm_factory):
    """非 symptom 类(disease/drug/anatomy)实体不进 confirmed/denied。"""
    from src.agent.nodes.build_query import build_query

    _setup_llm_mocks(
        mock_llm_factory,
        ner_entities=[
            NEREntity(text="阿莫西林", entity_type="drug", negation=False),
            NEREntity(text="右上腹", entity_type="anatomy", negation=False),
            NEREntity(text="腹痛", entity_type="symptom", negation=False),
        ],
        qc_dense="x",
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.chief_complaint = "右上腹痛"
    update = build_query(s)
    assert update["confirmed_symptoms"] == ["腹痛"]  # 只 symptom 进
    assert update["denied_symptoms"] == []


@patch("src.agent.nodes.build_query.get_llm")
def test_sparse_multifield_picks_up_slots_and_report(mock_llm_factory):
    """Step 2 多字段直采:slots 单值 + list + report positive/impressions 都进 sparse_queries。"""
    from src.agent.nodes.build_query import build_query

    _setup_llm_mocks(
        mock_llm_factory,
        ner_entities=[],
        qc_dense="x",
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.chief_complaint = "腹痛 3 天"
    s.present_illness_slots.trigger = ["进食后"]  # 2026-05-22:trigger 改 list[str]
    s.present_illness_slots.location = "右上腹"
    s.present_illness_slots.aggravating = ["油腻饮食"]
    s.report_findings = [
        {
            "report_type": "blood_routine",
            "positive_findings": ["白细胞升高"],
            "impressions": ["急性炎症"],
        },
        {
            "report_type": "imaging",
            "positive_findings": [],
            "impressions": ["胆囊壁正常 (-)"],  # 阴性应被过滤
        },
    ]
    update = build_query(s)
    sq = update["sparse_queries"]
    assert "腹痛 3 天" in sq
    assert "进食后" in sq
    assert "右上腹" in sq
    assert "油腻饮食" in sq
    assert "白细胞升高" in sq
    assert "急性炎症" in sq
    # 阴性 impression 过滤
    assert all("(-)" not in item for item in sq)
