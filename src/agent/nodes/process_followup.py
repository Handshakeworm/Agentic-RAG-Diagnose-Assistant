"""src/agent/nodes/process_followup.py — Agent ⑦ process_followup_answer(DEV_SPEC §4.1.2 ⑦)。

LLM 解析患者回答 → 维度级槽位回填 + 新症状提取。followup_round += 1 后回到
build_query 复跑流水线。

④ 已重设计为只产 slot / open 两类追问,⑦ 不再有"症状级 yes/no 回答分流"分支 —
开放式追问得到的新症状统一进 `new_symptoms` 字段,本节点直接 append 到 confirmed_symptoms。

中安全等级:失败 → 抛异常终止会话(回答未解析将导致信息丢失,不能静默)。
"""
from __future__ import annotations

import logging
import time

from src.agent.schemas.followup import FollowupParseResult
from src.agent.state import MedicalState, PresentIllnessSlots
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_followup_parse_prompt


_logger = logging.getLogger(__name__)
_NODE = "process_followup_answer"
_SCHEMA = "FollowupParseResult"

_MULTI_VALUE_SLOTS = {"aggravating", "relieving", "associated_symptoms"}

# spec §4.1.2 ⑦:维度槽位 → 自然语言追加片段(避免机器格式 k=v 拉低下轮 dense_query 改写质量)
_SLOT_PHRASES: dict[str, str] = {
    "onset_time": "起病于{value}",
    "onset_mode": "起病方式为{value}",
    "trigger": "诱因为{value}",
    "location": "部位在{value}",
    "nature": "性质为{value}",
    "severity": "程度为{value}",
    "duration_pattern": "时间规律为{value}",
    "aggravating": "{value}时加重",
    "relieving": "{value}时缓解",
    "associated_symptoms": "伴随{value}",
    "progression": "病程演变:{value}",
    "treatment_tried": "诊疗经过:{value}",
    "treatment_response": "治疗反应:{value}",
}


def _format_slot_addition(slot: str, value) -> str:
    """单个 slot 的自然语言片段;多值槽位 list 用顿号连接;未知 slot 兜底 'slot=value'。"""
    if isinstance(value, list):
        rendered = "、".join(str(v) for v in value if v)
    else:
        rendered = str(value) if value is not None else ""
    if not rendered:
        return ""
    template = _SLOT_PHRASES.get(slot)
    return template.format(value=rendered) if template else f"{slot}={rendered}"


def _apply_slot_fills(slots: PresentIllnessSlots, fills: dict) -> PresentIllnessSlots:
    """把 LLM 回填值套回 PresentIllnessSlots,类型不符时丢弃该项。"""
    data = slots.model_dump()
    for k, v in fills.items():
        if k not in data:
            _logger.warning("LLM returned unknown slot '%s', ignoring", k)
            continue
        if k in _MULTI_VALUE_SLOTS:
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                _logger.warning("slot '%s' expects list, got %r", k, v)
                continue
            existing = data[k] or []
            data[k] = list(dict.fromkeys(existing + v))  # 去重保留顺序
        else:
            if isinstance(v, list):
                v = "; ".join(map(str, v))
            data[k] = str(v) if v is not None else None
    return PresentIllnessSlots(**data)


def process_followup_answer(state: MedicalState) -> dict:
    """三类输出 + 槽位回填 + present_illness 追加 + followup_round +=1。"""
    prompt = build_followup_parse_prompt(
        followup_question=state.followup_question,
        followup_answer=state.followup_answer,
        questions=state.followup_questions,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(FollowupParseResult, method="json_mode").with_retry(stop_after_attempt=3)
        result: FollowupParseResult = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA},
            },
        )
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _logger.error("[%s] structured output failed: %s", _NODE, e, exc_info=True)
        raise  # 中安全:抛回 graph
    finally:
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(
            time.perf_counter() - t0
        )

    confirmed = list(state.confirmed_symptoms)
    denied = list(state.denied_symptoms)
    uncertain = list(state.uncertain_symptoms)

    # spec §4.1.2 ⑦:开放式追问的新症状 + 患者顺带补充的副症状,统一进 confirmed_symptoms。
    # ④ 已重设计为只产 slot / open 两类追问,无 yes/no 症状级分流。
    already_known = set(confirmed) | set(denied) | set(uncertain)
    for term in result.new_symptoms:
        if term and term not in already_known:
            confirmed.append(term)
            already_known.add(term)

    new_slots = _apply_slot_fills(state.present_illness_slots, result.slot_fills)

    # present_illness 追加新维度信息(spec §4.1.2 ⑦):自然语言句式拼接,
    # 避免机器格式 k=v 拉低下轮 build_query LLM 改写 dense_query 的质量
    appended = state.present_illness or ""
    if result.slot_fills:
        phrases = [
            _format_slot_addition(k, v) for k, v in result.slot_fills.items()
        ]
        addition = "；".join(p for p in phrases if p)
        if addition:
            appended = (appended + "  " + addition).strip()

    return {
        "confirmed_symptoms": confirmed,
        "denied_symptoms": denied,
        "uncertain_symptoms": uncertain,
        "present_illness_slots": new_slots,
        "present_illness": appended,
        "followup_round": state.followup_round + 1,
    }
