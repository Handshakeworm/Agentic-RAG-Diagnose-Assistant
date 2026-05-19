"""tests/unit/test_node_select_symptom.py — F6 ④ select_discriminative_symptom 单元测试。

④ 重设计后只剩 1 处 LLM 调用,LLM 同时出:追问项(questions)+ unaskable 粗筛
(unaskable_symptoms)。Mock LLM 返回 SmartFollowupOutput,验证主入口边界:
- LLM 出 slot + open 混合 → followup_questions 正确转储
- LLM 出空 questions → followup_questions 为空(信息已足,跳诊断)
- LLM 失败 → 兜底空 followup + 空 unaskable,不抛异常
- slot type 但 slot 字段缺失 → 该条丢弃
- unaskable 转储 → state.unaskable_symptoms 为 LLM 输出的 dict 列表
"""
from __future__ import annotations

from unittest.mock import patch

from src.agent.schemas.symptom_selection import (
    FollowupQuestion,
    SmartFollowupOutput,
    UnaskableSymptom,
)
from src.agent.state import create_initial_state


def _state_with_chief(chief: str = "腹痛", **kwargs):
    s = create_initial_state(patient_id="P", patient_input="x")
    s.chief_complaint = chief
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


@patch("src.agent.nodes.select_symptom.get_llm")
def test_slot_and_open_mixed(mock_llm):
    """LLM 出 slot + open 混合 → followup_questions 完整转储。"""
    from src.agent.nodes.select_symptom import select_discriminative_symptom

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = SmartFollowupOutput(
        questions=[
            FollowupQuestion(type="slot", slot="trigger"),
            FollowupQuestion(type="slot", slot="location"),
            FollowupQuestion(type="open", slot=None),
        ]
    )

    s = _state_with_chief()
    update = select_discriminative_symptom(s)

    fq = update["followup_questions"]
    assert len(fq) == 3
    types = [q["type"] for q in fq]
    assert types.count("slot") == 2
    assert types.count("open") == 1
    # slot type 应带 slot 字段;open type 不应有
    slot_items = [q for q in fq if q["type"] == "slot"]
    assert all("slot" in q and q["slot"] for q in slot_items)
    open_items = [q for q in fq if q["type"] == "open"]
    assert all("slot" not in q for q in open_items)


@patch("src.agent.nodes.select_symptom.get_llm")
def test_empty_questions_means_info_sufficient(mock_llm):
    """LLM 出空 questions + 空 unaskable → 都为空,路由会跳诊断。"""
    from src.agent.nodes.select_symptom import select_discriminative_symptom

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = SmartFollowupOutput(questions=[], unaskable_symptoms=[])

    s = _state_with_chief()
    update = select_discriminative_symptom(s)
    assert update["followup_questions"] == []
    assert update["unaskable_symptoms"] == []
    assert "info_gain" not in update  # info_gain 字段已随信息增益机制移除


@patch("src.agent.nodes.select_symptom.get_llm")
def test_unaskable_symptoms_passes_through(mock_llm):
    """LLM 输出的 unaskable_symptoms 应原样转 dict 写入 state。

    后续 ⑩ 会基于诊断结果再次精筛覆盖,这一步只验证 ④ 的转储不丢/不改字段。
    """
    from src.agent.nodes.select_symptom import select_discriminative_symptom

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = SmartFollowupOutput(
        questions=[FollowupQuestion(type="slot", slot="trigger")],
        unaskable_symptoms=[
            UnaskableSymptom(
                description="腹部 B 超提示有无胆囊壁增厚",
                reason="鉴别胆囊炎 vs 胃炎",
            ),
            UnaskableSymptom(
                description="血常规白细胞分类",
                reason="判断有无感染",
            ),
        ],
    )

    s = _state_with_chief()
    update = select_discriminative_symptom(s)
    assert len(update["unaskable_symptoms"]) == 2
    assert update["unaskable_symptoms"][0] == {
        "description": "腹部 B 超提示有无胆囊壁增厚",
        "reason": "鉴别胆囊炎 vs 胃炎",
    }
    assert update["unaskable_symptoms"][1]["description"] == "血常规白细胞分类"


@patch("src.agent.nodes.select_symptom.get_llm")
def test_slot_type_missing_slot_name_dropped(mock_llm):
    """slot type 但 slot 字段空 → 该条丢弃。"""
    from src.agent.nodes.select_symptom import select_discriminative_symptom

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = SmartFollowupOutput(
        questions=[
            FollowupQuestion(type="slot", slot=None),  # 缺 slot 名应丢弃
            FollowupQuestion(type="slot", slot="trigger"),
            FollowupQuestion(type="open", slot=None),
        ]
    )

    s = _state_with_chief()
    update = select_discriminative_symptom(s)
    # 第一条被丢弃,剩 2 条
    assert len(update["followup_questions"]) == 2
    assert update["followup_questions"][0] == {"type": "slot", "slot": "trigger"}
    assert update["followup_questions"][1] == {"type": "open"}


@patch("src.agent.nodes.select_symptom.get_llm")
def test_llm_failure_falls_back_to_empty(mock_llm):
    """LLM 失败 → 兜底空 followup,不抛异常(中安全等级失败处理)。"""
    from src.agent.nodes.select_symptom import select_discriminative_symptom

    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = ValueError("schema rejected")

    s = _state_with_chief()
    update = select_discriminative_symptom(s)
    assert update["followup_questions"] == []


@patch("src.agent.nodes.select_symptom.get_llm")
def test_questions_capped_at_max(mock_llm):
    """LLM 返回超 quota 的 questions → 截到 MAX_FOLLOWUP_QUESTIONS。"""
    from config.settings import settings
    from src.agent.nodes.select_symptom import select_discriminative_symptom

    K = settings.agent_limits.MAX_FOLLOWUP_QUESTIONS
    # 构造 K+2 个有效条目(LLM schema 已经 max_length=5,这里测兜底截断)
    # SmartFollowupOutput 自身 max_length=5,所以直接用 K 条就够
    mock_chain = mock_llm.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = SmartFollowupOutput(
        questions=[FollowupQuestion(type="slot", slot=f"slot_{i}") for i in range(K)]
    )

    s = _state_with_chief()
    update = select_discriminative_symptom(s)
    assert len(update["followup_questions"]) <= K
