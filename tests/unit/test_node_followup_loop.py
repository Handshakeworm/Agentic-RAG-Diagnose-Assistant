"""tests/unit/test_node_followup_loop.py — F8 追问循环单元测试。

⑤ generate_followup(自由文本)+ ⑥ wait_followup_answer(interrupt)+ ⑦
process_followup_answer(structured)。

④ 重设计后,⑦ 只处理两种 followup type:slot(维度回填)+ open(新症状回填)。
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.agent.schemas.followup import FollowupParseResult
from src.agent.state import create_initial_state


# ────────────────────────────────────────────────────────────────────────────
# ⑤ generate_followup
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.generate_followup.get_llm")
def test_generate_followup_writes_question(mock_llm_factory):
    from src.agent.nodes.generate_followup import generate_followup

    mock_chain = mock_llm_factory.return_value.with_retry.return_value
    msg = MagicMock()
    msg.content = "请问您腹痛是什么时候开始的?除此之外还有别的不舒服吗?"
    mock_chain.invoke.return_value = msg

    s = create_initial_state(patient_id="P", patient_input="x")
    s.chief_complaint = "腹痛"
    s.followup_questions = [
        {"slot": "onset_time", "type": "slot"},
        {"type": "open"},
    ]
    update = generate_followup(s)
    assert "followup_question" in update
    assert "什么时候" in update["followup_question"] or "别的不舒服" in update["followup_question"]


@patch("src.agent.nodes.generate_followup.get_llm")
def test_generate_followup_empty_questions_returns_empty_string(mock_llm_factory):
    """followup_questions 为空 → 透传,不调 LLM。"""
    from src.agent.nodes.generate_followup import generate_followup

    s = create_initial_state(patient_id="P", patient_input="x")
    update = generate_followup(s)
    assert update == {"followup_question": ""}
    mock_llm_factory.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────
# ⑥ wait_followup_answer
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.wait_followup_answer.interrupt")
def test_wait_followup_answer_calls_interrupt_with_question(mock_interrupt):
    from src.agent.nodes.wait_followup_answer import wait_followup_answer

    mock_interrupt.return_value = "我有反酸"
    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_question = "请问您有反酸吗?"
    update = wait_followup_answer(s)
    mock_interrupt.assert_called_once_with("请问您有反酸吗?")
    assert update == {"followup_answer": "我有反酸"}


# ────────────────────────────────────────────────────────────────────────────
# ⑦ process_followup_answer — 只处理 slot_fills + new_symptoms
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.process_followup.get_llm")
def test_process_followup_slot_fills_and_new_symptoms(mock_llm_factory):
    """slot 类追问回填维度,open 类追问产生新症状进 confirmed_symptoms。"""
    from src.agent.nodes.process_followup import process_followup_answer

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = FollowupParseResult(
        slot_fills={"trigger": "进食后", "aggravating": ["饥饿"]},
        new_symptoms=["反酸", "夜间盗汗"],
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_question = "..."
    s.followup_answer = "吃饱了就疼,饿了也疼,还有反酸和盗汗"
    s.followup_questions = [
        {"slot": "trigger", "type": "slot"},
        {"slot": "aggravating", "type": "slot"},
        {"type": "open"},
    ]
    update = process_followup_answer(s)

    # 维度回填
    assert update["present_illness_slots"].trigger == "进食后"
    assert "饥饿" in update["present_illness_slots"].aggravating
    # 新症状直接进 confirmed
    assert "反酸" in update["confirmed_symptoms"]
    assert "夜间盗汗" in update["confirmed_symptoms"]
    assert update["followup_round"] == 1


@patch("src.agent.nodes.process_followup.get_llm")
def test_process_followup_new_symptoms_dedup(mock_llm_factory):
    """new_symptoms 里已在 confirmed/denied/uncertain 的不重复追加。"""
    from src.agent.nodes.process_followup import process_followup_answer

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = FollowupParseResult(
        slot_fills={},
        new_symptoms=["反酸", "腹胀"],  # 反酸已 confirmed,腹胀新增
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_answer = "x"
    s.followup_questions = [{"type": "open"}]
    s.confirmed_symptoms = ["反酸"]
    update = process_followup_answer(s)

    # 反酸不重复(只出现一次)
    assert update["confirmed_symptoms"].count("反酸") == 1
    assert "腹胀" in update["confirmed_symptoms"]


@patch("src.agent.nodes.process_followup.get_llm")
def test_process_followup_llm_failure_raises(mock_llm_factory):
    """中安全等级:失败必须抛异常,不能静默吃掉患者回答。"""
    from src.agent.nodes.process_followup import process_followup_answer

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = ValueError("schema rejected")

    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_answer = "有反酸"
    s.followup_questions = [{"type": "open"}]
    with pytest.raises(ValueError):
        process_followup_answer(s)
