"""tests/unit/test_node_followup_loop.py — F8 追问循环单元测试。

⑤ generate_followup(2026-05-21 起单 LLM 双 list 出口:检索前 holistic gate)+
⑥ wait_followup_answer(interrupt)+ ⑦ process_followup_answer(structured)。

⑤ 当前契约:1 次 LLM 调用产 `HolisticGateOutput { askable_questions, unaskable_findings }`,
按"患者能不能答"分两路;优先级 askable > unaskable > 空;每次都写
`unaskable_symptoms` 供下游消费。④ 路径 → ⑥ 直连不经过 ⑤。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.schemas.followup import FollowupParseResult
from src.agent.schemas.intake import HolisticGateOutput, TargetedFollowupItem
from src.agent.schemas.symptom_selection import UnaskableSymptom
from src.agent.state import create_initial_state


# ────────────────────────────────────────────────────────────────────────────
# ⑤ generate_followup — 检索前 holistic gate(单 LLM, 双 list 出口)
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.generate_followup.get_llm")
def test_generate_followup_askable_writes_questions_and_text(mock_llm_factory):
    """LLM 出 askable 题 → followup_questions 写入 + followup_question 模板拼好 + 尾部加 open。

    优先级:askable 非空 → 走 ⑥(不写 ⑧a 出口),但 unaskable_symptoms 仍要写下来供下游用。
    """
    from src.agent.nodes.generate_followup import generate_followup

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = HolisticGateOutput(
        askable_questions=[
            TargetedFollowupItem(question="疼痛是不是放射到右肩胛?", target="radiation"),
        ],
        unaskable_findings=[
            UnaskableSymptom(description="Murphy 征查体", reason="鉴别胆囊炎"),
        ],
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.chief_complaint = "腹痛"
    update = generate_followup(s)

    # askable 优先走 ⑥
    assert update["followup_questions"][0]["type"] == "targeted"
    assert update["followup_questions"][0]["question"] == "疼痛是不是放射到右肩胛?"
    # 尾部追加 open 兜底
    assert update["followup_questions"][-1] == {"type": "open"}
    assert "放射到右肩胛" in update["followup_question"]
    assert "别的地方不舒服" in update["followup_question"]
    # unaskable 也写下来(供下游 ⑩/⑧a 消费)
    assert update["unaskable_symptoms"] == [
        {"description": "Murphy 征查体", "reason": "鉴别胆囊炎"}
    ]


@patch("src.agent.nodes.generate_followup.get_llm")
def test_generate_followup_unaskable_only_routes_to_recommend_exam(mock_llm_factory):
    """无 askable 但有 unaskable → followup_questions 空 + unaskable_symptoms 写入,
    router 据此走 to_recommend_exam → ⑧a 首诊推单。"""
    from src.agent.nodes.generate_followup import generate_followup

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = HolisticGateOutput(
        askable_questions=[],
        unaskable_findings=[
            UnaskableSymptom(description="腹部 B 超", reason="排查胆石/胆囊壁"),
            UnaskableSymptom(description="血常规", reason="判断感染"),
        ],
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    update = generate_followup(s)
    assert update["followup_questions"] == []
    assert update["followup_question"] == ""
    assert len(update["unaskable_symptoms"]) == 2
    assert update["unaskable_symptoms"][0]["description"] == "腹部 B 超"


@patch("src.agent.nodes.generate_followup.get_llm")
def test_generate_followup_both_empty_routes_to_retrieval(mock_llm_factory):
    """两个 list 都空 → 节点返三空,router 走 to_build_query 进检索。"""
    from src.agent.nodes.generate_followup import generate_followup

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = HolisticGateOutput(askable_questions=[], unaskable_findings=[])

    s = create_initial_state(patient_id="P", patient_input="x")
    update = generate_followup(s)
    assert update == {
        "followup_questions": [],
        "followup_question": "",
        "unaskable_symptoms": [],
    }


@patch("src.agent.nodes.generate_followup.get_llm")
def test_generate_followup_llm_failure_falls_back_to_empty(mock_llm_factory):
    """中安全等级:LLM 失败 → 兜底返两 list 都空(让 router 走 ② 进检索,不阻塞流水线)。"""
    from src.agent.nodes.generate_followup import generate_followup

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = ValueError("schema rejected")

    s = create_initial_state(patient_id="P", patient_input="x")
    update = generate_followup(s)
    assert update == {
        "followup_questions": [],
        "followup_question": "",
        "unaskable_symptoms": [],
    }


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


# ────────────────────────────────────────────────────────────────────────────
# obstetric_fills 回写 state.medical_history.obstetric_history
# (⑪ safety_gate 妊娠/哺乳禁忌兜底硬依赖)
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.process_followup.get_llm")
def test_process_followup_obstetric_pregnant_writes_history(mock_llm_factory):
    """obstetric_fills.is_pregnant=true → 写回 medical_history.obstetric_history,
    pregnancy_status='pregnant' 供 ⑪ safety_gate 规则层读取。"""
    from src.agent.nodes.process_followup import process_followup_answer

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = FollowupParseResult(
        obstetric_fills={"is_pregnant": True, "is_lactating": False},
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_answer = "怀孕 3 个月了,没在哺乳"
    s.followup_questions = [{"type": "obstetric"}]
    s.medical_history = {"obstetric_history": None}

    update = process_followup_answer(s)
    ob = update["medical_history"]["obstetric_history"]
    assert ob["is_pregnant"] is True
    assert ob["pregnancy_status"] == "pregnant"  # ⑪ safety_gate 读这个字段
    assert ob["is_lactating"] is False
    assert ob["lactation_status"] == "not_pregnant" or ob["lactation_status"] == "not_lactating"


@patch("src.agent.nodes.process_followup.get_llm")
def test_process_followup_no_obstetric_question_preserves_history(mock_llm_factory):
    """无 obstetric 追问 → medical_history.obstetric_history 不变(透传)。"""
    from src.agent.nodes.process_followup import process_followup_answer

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = FollowupParseResult(
        slot_fills={"trigger": "进食"},
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_answer = "饭后疼"
    s.followup_questions = [{"slot": "trigger", "type": "slot"}]
    existing_history = {"obstetric_history": None, "basic_info": {"gender": "male"}}
    s.medical_history = existing_history

    update = process_followup_answer(s)
    assert update["medical_history"] == existing_history


@patch("src.agent.nodes.process_followup.get_llm")
def test_process_followup_obstetric_null_answer_does_not_write(mock_llm_factory):
    """回答不明 → obstetric_fills 两 key 都 null → 不写回(避免污染档案)。"""
    from src.agent.nodes.process_followup import process_followup_answer

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = FollowupParseResult(
        obstetric_fills={"is_pregnant": None, "is_lactating": None},
    )

    s = create_initial_state(patient_id="P", patient_input="x")
    s.followup_answer = "不知道"
    s.followup_questions = [{"type": "obstetric"}]
    s.medical_history = {"obstetric_history": None}

    update = process_followup_answer(s)
    # 两 key 都 null → 不应该 fabricate obstetric_history
    assert update["medical_history"].get("obstetric_history") is None
