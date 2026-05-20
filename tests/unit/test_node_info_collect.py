"""tests/unit/test_node_info_collect.py — F2 ① info_collect 单元测试(DEV_SPEC §4.1.2 ①)。

Mock LLM(返回结构化 schema 实例);验证:
- 完整输入 → 13 维 slots 全填(无空槽)
- 简短输入 → 多空槽保持 None / []
- LLM 失败 → 中安全等级,抛异常,_failures 计数 +1
- **不写 medical_history**(由 ⓪a load + ⑦ 增量合并,① 写就覆盖增量了)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.schemas.info_collect import (
    InfoCollectOutput,
    PresentIllnessSlots as SchemaSlots,
)
from src.agent.state import MedicalState, PresentIllnessSlots, create_initial_state


def _make_state(patient_input: str) -> MedicalState:
    return create_initial_state(patient_id="P-TEST", patient_input=patient_input)


@patch("src.agent.nodes.info_collect.get_llm")
def test_full_input_fills_all_slots(mock_llm_factory):
    """完整输入应让 13 维 slots 全部有值;update 不含 medical_history(防覆盖)。"""
    from src.agent.nodes.info_collect import info_collect

    schema_slots = SchemaSlots(
        onset_time="3天前",
        onset_mode="急性",
        trigger="进食后",
        location="上腹",
        nature="胀痛",
        severity="中",
        duration_pattern="持续性",
        aggravating=["进食", "饥饿"],
        relieving=["热敷"],
        associated_symptoms=["反酸"],
        progression="加重",
        treatment_tried="服用奥美拉唑",
        treatment_response="部分缓解",
    )
    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = InfoCollectOutput(
        chief_complaint="上腹痛3天",
        present_illness="3天前进食后开始上腹胀痛,持续性,服奥美拉唑部分缓解",
        present_illness_slots=schema_slots,
    )

    state = _make_state(
        "我3天前吃完饭后开始上腹胀痛,一直持续没断过,吃了奥美拉唑稍微好点,还有反酸"
    )
    update = info_collect(state)

    assert update["chief_complaint"] == "上腹痛3天"
    slots = update["present_illness_slots"]
    assert isinstance(slots, PresentIllnessSlots)
    assert slots.onset_time == "3天前"
    assert slots.aggravating == ["进食", "饥饿"]
    assert slots.associated_symptoms == ["反酸"]
    assert slots.treatment_response == "部分缓解"

    # ① 合并了 ⑦ 在 to_info_collect 路径上的活:LLM 没返 history/obstetric/new_symptoms
    # 时,medical_history merge 后等于 state 原值(此处空 dict),confirmed_symptoms 不新增
    assert update["medical_history"] == {}
    assert update["confirmed_symptoms"] == []


@patch("src.agent.nodes.info_collect.get_llm")
def test_short_input_leaves_slots_empty(mock_llm_factory):
    """简短输入对应多空槽,LLM 应保持未提及维度为 None / []。"""
    from src.agent.nodes.info_collect import info_collect

    sparse_slots = SchemaSlots(location="头部")  # 其余全保持默认 None / []
    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = InfoCollectOutput(
        chief_complaint="头痛",
        present_illness="头痛,未述其他",
        present_illness_slots=sparse_slots,
    )

    update = info_collect(_make_state("头疼"))

    slots = update["present_illness_slots"]
    assert slots.location == "头部"
    assert slots.onset_time is None
    assert slots.trigger is None
    assert slots.aggravating == []
    assert slots.associated_symptoms == []


@patch("src.agent.nodes.info_collect.get_llm")
def test_llm_failure_raises_and_increments_failure_metric(mock_llm_factory):
    """中安全等级:重试耗尽后抛异常,_failures 计数 +1(spec §9.1 模板)。"""
    from src.agent.nodes.info_collect import info_collect
    from src.common.metrics import _failures

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = ValueError("structured output rejected")

    before = _failures.labels(
        node="info_collect_step1", schema="InfoCollectOutput", exception_type="ValueError"
    )._value.get()

    with pytest.raises(ValueError):
        info_collect(_make_state("胃疼"))

    after = _failures.labels(
        node="info_collect_step1", schema="InfoCollectOutput", exception_type="ValueError"
    )._value.get()
    assert after == before + 1
