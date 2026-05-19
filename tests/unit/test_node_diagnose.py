"""tests/unit/test_node_diagnose.py — F10 ⑩ diagnose 单元测试(DEV_SPEC §4.1.2 ⑩)。

⑩ 重设计为 1 步 LLM 后,路径精简到 4 条(原 5 条 step_1/2/3_failed 合并成 step_1_failed):
1. followup_round 触顶 → failure_reason == "followup_round_capped"
2. 正常 1 步 LLM 成功 → failure_reason is None + retained_unaskable 写回
3. LLM 失败 → failure_reason.startswith("step_1_structured_output_failed")
4. vision LLM 路由验证(spec §3.2.3 + §9.3)

兜底场景共同断言:differentiation_type == "insufficient" 且 probability == 0.0,
last_diagnose_prompt / raw_output 已写入(供审计)。
"""
from __future__ import annotations

from unittest.mock import patch

from config.settings import settings
from src.agent.schemas.diagnosis import DiagnosisOutput, RankedDisease
from src.agent.schemas.symptom_selection import UnaskableSymptom
from src.agent.state import create_initial_state


def _state_for_diagnose():
    s = create_initial_state(patient_id="P", patient_input="x")
    s.chief_complaint = "腹痛"
    s.confirmed_symptoms = ["腹痛"]
    s.candidate_chunks = [
        {
            "source_chunk_id": "c1",
            "rrf_score": 0.1,
            "vector_hits": [
                {"vector_type": "original", "rank": 1, "matched_text": "胆囊炎症状"}
            ],
        }
    ]
    return s


def _ok_output(retained_unaskable: list[UnaskableSymptom] | None = None):
    return DiagnosisOutput(
        results=[
            RankedDisease(
                disease="胆囊炎",
                probability=0.7,
                evidence=["症状典型", "右上腹压痛"],
                differentiation="与胃溃疡相比,胆囊炎多有 Murphy 征阳性",
                differentiation_type="confirmed",
            )
        ],
        retained_unaskable=retained_unaskable or [],
    )


# ────────────────────────────────────────────────────────────────────────────
# 路径 1:Step -1 触顶兜底
# ────────────────────────────────────────────────────────────────────────────


def test_followup_round_cap_short_circuits():
    from src.agent.nodes.diagnose import diagnose

    s = _state_for_diagnose()
    s.followup_round = settings.agent_limits.MAX_FOLLOWUP_ROUNDS
    update = diagnose(s)

    res = update["diagnosis_result"][0]
    assert res["failure_reason"] == "followup_round_capped"
    assert res["differentiation_type"] == "insufficient"
    assert res["probability"] == 0.0


# ────────────────────────────────────────────────────────────────────────────
# 路径 2:正常 1 步 LLM 成功
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.rerank_with_fallback", return_value=[0])
@patch(
    "src.agent.nodes.diagnose.lookup_chunk_content",
    return_value={"c1": {"chunk_raw_text": "胆囊炎诊断标准...", "parent_chunk_id": None}},
)
@patch("src.agent.nodes.diagnose.get_llm")
def test_normal_single_step_succeeds(mock_llm, _lookup, _rerank):
    from src.agent.nodes.diagnose import diagnose

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = _ok_output()

    s = _state_for_diagnose()
    update = diagnose(s)

    res = update["diagnosis_result"][0]
    assert res["failure_reason"] is None
    assert res["disease"] == "胆囊炎"
    assert "last_reranked_chunks" in update
    # 正常路径不写 last_diagnose_prompt / raw_output
    assert "last_diagnose_prompt" not in update
    assert "last_diagnose_raw_output" not in update
    # 正常路径必须写回 unaskable_symptoms(覆盖 ④ 粗筛);本 case LLM 返空 = 清空
    assert update["unaskable_symptoms"] == []


# ────────────────────────────────────────────────────────────────────────────
# retained_unaskable 写回:覆盖 ④ 粗筛
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.rerank_with_fallback", return_value=[0])
@patch(
    "src.agent.nodes.diagnose.lookup_chunk_content",
    return_value={"c1": {"chunk_raw_text": "x", "parent_chunk_id": None}},
)
@patch("src.agent.nodes.diagnose.get_llm")
def test_retained_unaskable_overwrites_state(mock_llm, _lookup, _rerank):
    """⑩ 出 retained_unaskable → 覆盖 state.unaskable_symptoms(粗筛 → 精筛)。"""
    from src.agent.nodes.diagnose import diagnose

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = DiagnosisOutput(
        results=[
            RankedDisease(
                disease="胆囊炎",
                probability=0.55,
                evidence=["症状部分匹配"],
                differentiation=None,
                differentiation_type="need_exam",
            )
        ],
        retained_unaskable=[
            UnaskableSymptom(
                description="腹部 B 超确认胆囊壁厚度 + 有无结石",
                reason="鉴别胆囊炎 vs 胃炎",
            ),
        ],
    )

    s = _state_for_diagnose()
    # 模拟 ④ 已经写过粗筛(2 条),验证 ⑩ 输出会**覆盖**这份
    s.unaskable_symptoms = [
        {"description": "肝功能", "reason": "粗筛随便填的"},
        {"description": "胰酶", "reason": "粗筛随便填的"},
    ]
    update = diagnose(s)

    assert len(update["unaskable_symptoms"]) == 1
    assert update["unaskable_symptoms"][0]["description"] == "腹部 B 超确认胆囊壁厚度 + 有无结石"
    assert update["unaskable_symptoms"][0]["reason"] == "鉴别胆囊炎 vs 胃炎"


# ────────────────────────────────────────────────────────────────────────────
# 路径 3:LLM 失败 → step_1_structured_output_failed 兜底
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.rerank_with_fallback", return_value=[0])
@patch(
    "src.agent.nodes.diagnose.lookup_chunk_content",
    return_value={"c1": {"chunk_raw_text": "x", "parent_chunk_id": None}},
)
@patch("src.agent.nodes.diagnose.get_llm")
def test_llm_failure_yields_insufficient(mock_llm, _lookup, _rerank):
    from src.agent.nodes.diagnose import diagnose

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = ValueError("schema rejected")

    s = _state_for_diagnose()
    update = diagnose(s)
    res = update["diagnosis_result"][0]
    assert res["failure_reason"].startswith("step_1_structured_output_failed")
    assert res["differentiation_type"] == "insufficient"
    assert res["probability"] == 0.0
    assert update["last_diagnose_prompt"] is not None
    assert update["last_diagnose_raw_output"] is not None


# ────────────────────────────────────────────────────────────────────────────
# 路径 4:vision LLM 路由(spec §3.2.3 LLM 路由 + §9.3 diagnose 行)
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.rerank_with_fallback", return_value=[0])
@patch(
    "src.agent.nodes.diagnose.lookup_chunk_content",
    return_value={"c1": {"chunk_raw_text": "x", "parent_chunk_id": None}},
)
@patch("src.agent.nodes.diagnose.get_llm")
def test_uses_vision_llm(mock_llm, _lookup, _rerank):
    """spec §3.2.3:⑩ 1 步 LLM 走 vision LLM(原生多模态,figure 截图作 image_url)。"""
    from src.agent.nodes.diagnose import diagnose

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = _ok_output()

    s = _state_for_diagnose()
    diagnose(s)

    # 验证 get_llm 被调用,且传 vision 三件套
    assert mock_llm.called
    call = mock_llm.call_args
    assert call.kwargs.get("model") == settings.llm.VISION_MODEL_NAME
    assert call.kwargs.get("base_url") == settings.llm.VISION_BASE_URL
