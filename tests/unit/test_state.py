"""tests/unit/test_state.py — 锁住 MedicalState (Pydantic) 与 create_initial_state(DEV_SPEC §4.1.1 / §4.1.1a)。

State 一旦上线就被 16 节点 + 2 路由器读写,任何字段缺失/类型漂移都会引发 ValidationError 或诊断逻辑跑偏。
这些测试把 spec 文本契约钉死在断言里:
- §4.1.1 字段清单(37 个)+ Pydantic BaseModel 形态
- §4.1.1a 初始值表
- 13 维 PresentIllnessSlots(3 list[str] + 10 str | None,且自身是 BaseModel)
- SessionTokenUsage / SessionLatencyMs 嵌套 BaseModel
- 多 session 不共享可变对象(避免 list 别名 bug)
- §9.2 schema 演化:加字段必须有默认 + 类型校验 immediate ValidationError
"""

from __future__ import annotations

import pytest


def test_create_initial_state_returns_all_35_fields() -> None:
    """§4.1.1 字段清单完整性 — 35 个字段一个不能少(intake_phase 加入)。"""
    from src.agent.state import MedicalState, create_initial_state

    state = create_initial_state(patient_id="P001", patient_input="头疼一周")
    assert isinstance(state, MedicalState)

    expected_fields = {
        # 消息历史
        "messages",
        # 患者信息
        "patient_id", "patient_input", "chief_complaint", "present_illness",
        "present_illness_slots", "medical_history", "exam_reports", "report_findings",
        # 召回与候选(extracted_symptoms 随 ④ 节点废弃一并移除;standardized_entities 随 EL 移除)
        "dense_query", "sparse_queries", "candidate_chunks",
        "confirmed_symptoms", "denied_symptoms", "uncertain_symptoms",
        # 追问控制(info_gain 随信息增益机制移除一并删除;intake_phase 新增)
        "followup_round", "last_nlu_round", "followup_question", "followup_answer",
        "followup_questions", "unaskable_symptoms",
        "exam_round", "pending_exam_results",
        "intake_phase", "followup_source",
        # 诊断
        "diagnosis_result",
        # 安全约束
        "safety_constraints",
        # 建议输出
        "recommended_tests", "medication_advice", "risk_warnings", "final_response",
        # 审计埋点(§9.6)
        "last_reranked_chunks", "session_token_usage", "session_latency_ms",
        "last_diagnose_prompt", "last_diagnose_raw_output",
    }
    actual_fields = set(MedicalState.model_fields.keys())
    missing = expected_fields - actual_fields
    extra = actual_fields - expected_fields
    assert not missing, f"缺失字段: {missing}"
    assert not extra, f"多余字段: {extra}"
    assert len(actual_fields) == 36


def test_caller_required_fields_passed_through() -> None:
    """patient_id / patient_input 必须原样保留(不被任何默认值覆盖)。"""
    from src.agent.state import create_initial_state

    state = create_initial_state(patient_id="P-2026-001", patient_input="发烧 3 天,T 39.2°C")
    assert state.patient_id == "P-2026-001"
    assert state.patient_input == "发烧 3 天,T 39.2°C"


def test_caller_required_fields_validated() -> None:
    """缺 patient_id 或 patient_input 必须 ValidationError(§9.2 加字段保护)。"""
    from pydantic import ValidationError

    from src.agent.state import MedicalState

    with pytest.raises(ValidationError):
        MedicalState()  # 缺两个必填


def test_present_illness_slots_13_dimensions_strongly_typed() -> None:
    """§4.1.1 现病史 13 维 — 现在是强类型 PresentIllnessSlots BaseModel,不再是裸 dict。"""
    from src.agent.state import PresentIllnessSlots, create_initial_state

    slots = create_initial_state(patient_id="P001", patient_input="x").present_illness_slots
    assert isinstance(slots, PresentIllnessSlots)

    multi_value = {"aggravating", "relieving", "associated_symptoms"}
    single_value = {
        "onset_time", "onset_mode", "trigger", "location", "nature",
        "severity", "duration_pattern", "progression",
        "treatment_tried", "treatment_response",
    }
    assert set(PresentIllnessSlots.model_fields.keys()) == multi_value | single_value
    assert len(PresentIllnessSlots.model_fields) == 13

    for k in multi_value:
        assert getattr(slots, k) == [], f"{k} 应为空 list"
    for k in single_value:
        assert getattr(slots, k) is None, f"{k} 应为 None"


def test_initial_values_match_spec_4_1_1a() -> None:
    """§4.1.1a 初始值表 — 每个字段的零值锁死。"""
    from src.agent.state import create_initial_state

    s = create_initial_state(patient_id="P001", patient_input="x")

    # 字符串零值
    assert s.chief_complaint == ""
    assert s.present_illness == ""
    assert s.dense_query == ""
    assert s.followup_question == ""
    assert s.followup_answer == ""
    assert s.final_response == ""

    # 整数零值
    assert s.followup_round == 0
    assert s.last_nlu_round == 0
    assert s.exam_round == 0

    # 空容器
    assert s.messages == []
    assert s.medical_history == {}
    assert s.safety_constraints == {}
    for key in [
        "exam_reports", "report_findings",
        "sparse_queries", "candidate_chunks",
        "confirmed_symptoms", "denied_symptoms", "uncertain_symptoms",
        "followup_questions", "unaskable_symptoms", "pending_exam_results",
        "diagnosis_result", "recommended_tests", "medication_advice",
        "risk_warnings", "last_reranked_chunks",
    ]:
        assert getattr(s, key) == [], f"{key} 初始应为 []"

    # 审计嵌套 BaseModel
    assert s.session_token_usage.prompt_tokens == 0
    assert s.session_token_usage.completion_tokens == 0
    assert s.session_token_usage.total_tokens == 0
    assert s.session_latency_ms.intent == 0
    assert s.session_latency_ms.retrieval == 0
    assert s.session_latency_ms.rerank == 0
    assert s.session_latency_ms.llm_call == 0
    assert s.session_latency_ms.post_process == 0

    # 兜底专用字段:正常路径保持 None
    assert s.last_diagnose_prompt is None
    assert s.last_diagnose_raw_output is None


def test_two_sessions_dont_share_mutable_objects() -> None:
    """两个 session 必须有独立的可变对象 — Field(default_factory=...) 保证。"""
    from src.agent.state import create_initial_state

    s1 = create_initial_state(patient_id="P001", patient_input="A")
    s2 = create_initial_state(patient_id="P002", patient_input="B")

    # 嵌套 BaseModel + 内部 list 都不能共享
    assert s1.present_illness_slots is not s2.present_illness_slots
    assert s1.present_illness_slots.aggravating is not s2.present_illness_slots.aggravating

    # 顶层 list 字段也不能共享
    assert s1.messages is not s2.messages
    assert s1.confirmed_symptoms is not s2.confirmed_symptoms
    assert s1.session_token_usage is not s2.session_token_usage

    # 实战:改 s1 不影响 s2
    s1.present_illness_slots.aggravating.append("进食")
    s1.confirmed_symptoms.append("腹痛")
    assert s2.present_illness_slots.aggravating == []
    assert s2.confirmed_symptoms == []


def test_type_validation_rejects_wrong_types() -> None:
    """§9.2 类型校验 — Pydantic 应在实例化时立即拦下错误类型,而不是默默接受到运行时崩。"""
    from pydantic import ValidationError

    from src.agent.state import MedicalState

    # confirmed_symptoms 期望 list[str],传 int 应拒绝
    with pytest.raises(ValidationError):
        MedicalState(patient_id="P001", patient_input="x", confirmed_symptoms=123)

    # followup_round 期望 int,传 "abc" 应拒绝(注:Pydantic 默认会尝试转换 "5" → 5,但 "abc" 转不了)
    with pytest.raises(ValidationError):
        MedicalState(patient_id="P001", patient_input="x", followup_round="abc")


def test_old_state_reload_fills_missing_fields_with_defaults() -> None:
    """§9.2 schema 演化关键能力 — 老 checkpointer 数据反序列化时,缺字段自动填默认。"""
    from src.agent.state import MedicalState

    # 模拟老 checkpointer 的 dump,只有调用方必填 + 部分字段
    old_dump = {
        "patient_id": "P-OLD-001",
        "patient_input": "肚子痛",
        "followup_round": 3,  # 老数据有这个
        # 没有 confirmed_symptoms / session_token_usage 等任何新字段
    }
    reloaded = MedicalState.model_validate(old_dump)
    assert reloaded.patient_id == "P-OLD-001"
    assert reloaded.followup_round == 3
    # 缺失字段被默认值填充,**没有 KeyError 或 ValidationError**
    assert reloaded.confirmed_symptoms == []
    assert reloaded.session_token_usage.prompt_tokens == 0
    assert reloaded.last_diagnose_prompt is None
