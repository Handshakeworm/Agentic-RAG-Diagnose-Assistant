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
from src.agent.state import SLOT_UNKNOWN_SENTINEL, MedicalState, PresentIllnessSlots
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
    """把 LLM 回填值套回 PresentIllnessSlots,类型不符时丢弃该项。

    哨兵语义:`SLOT_UNKNOWN_SENTINEL`("(患者未明确)")表示"已问无答"。多值槽合并时
    如果新值含真实内容,先把已存的哨兵剥掉,避免出现 ['(患者未明确)', '进食'] 这种
    既哨兵又真值的脏列表。
    """
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
            existing = list(data[k] or [])
            new_has_real = any(item and item != SLOT_UNKNOWN_SENTINEL for item in v)
            if new_has_real:
                # 患者从"不知道"升级到给出真实信息 → 剥掉旧哨兵
                existing = [x for x in existing if x != SLOT_UNKNOWN_SENTINEL]
                v = [x for x in v if x != SLOT_UNKNOWN_SENTINEL]
            data[k] = list(dict.fromkeys(existing + v))  # 去重保留顺序
        else:
            if isinstance(v, list):
                v = "; ".join(map(str, v))
            data[k] = str(v) if v is not None else None
    return PresentIllnessSlots(**data)


def _merge_obstetric_fills(
    medical_history: dict, fills: dict[str, bool | None] | None
) -> dict:
    """把 obstetric_fills 写回 medical_history['obstetric_history']。

    返回新 dict(LangGraph 状态不可变约定)。fills 为 None / 空 / 全 None → 透传不变。
    字符串值与 patient_repo.load_medical_history 对齐(pregnant/not_pregnant 等),
    供 ⑪ safety_gate 规则层直接判断。
    """
    if not fills:
        return medical_history
    new_hist = dict(medical_history)
    ob = dict(new_hist.get("obstetric_history") or {})
    if "is_pregnant" in fills and fills["is_pregnant"] is not None:
        ob["is_pregnant"] = fills["is_pregnant"]
        ob["pregnancy_status"] = "pregnant" if fills["is_pregnant"] else "not_pregnant"
    if "is_lactating" in fills and fills["is_lactating"] is not None:
        ob["is_lactating"] = fills["is_lactating"]
        ob["lactation_status"] = "lactating" if fills["is_lactating"] else "not_lactating"
    if ob:  # 至少有一项被写,才覆盖
        new_hist["obstetric_history"] = ob
    return new_hist


def _merge_history_fills(
    medical_history: dict, fills: dict[str, list[str]] | None
) -> dict:
    """把 history_fills 三段扁平 list 追加到 medical_history 对应表(allergy / medication / past)。

    字段映射对齐 `patient_repo.load_medical_history` 返回结构:
      - allergies        → allergy_history[].substance
      - medications      → medication_history[].drug_name(is_current=True)
      - past_conditions  → past_history.medical_history[].condition

    幂等:已存在的同名条目跳过(用 substance / drug_name / condition 主键比对)。
    fills 为 None → 透传;某段 list 为空 → 表示"已问无答",不追加但视为已询问。
    """
    if fills is None:
        return medical_history
    new_hist = dict(medical_history)

    # 过敏 — 主键: substance
    allergies_new = fills.get("allergies") or []
    if allergies_new:
        existing = list(new_hist.get("allergy_history") or [])
        seen = {r.get("substance") for r in existing if r.get("substance")}
        for name in allergies_new:
            if name and name not in seen:
                existing.append({"substance": name})
                seen.add(name)
        new_hist["allergy_history"] = existing

    # 用药 — 主键: drug_name
    meds_new = fills.get("medications") or []
    if meds_new:
        existing = list(new_hist.get("medication_history") or [])
        seen = {r.get("drug_name") for r in existing if r.get("drug_name")}
        for name in meds_new:
            if name and name not in seen:
                existing.append({"drug_name": name, "is_current": True})
                seen.add(name)
        new_hist["medication_history"] = existing

    # 既往疾病 — 嵌在 past_history.medical_history,主键: condition
    past_new = fills.get("past_conditions") or []
    if past_new:
        past_section = dict(new_hist.get("past_history") or {})
        past_list = list(past_section.get("medical_history") or [])
        seen = {r.get("condition") for r in past_list if r.get("condition")}
        for name in past_new:
            if name and name not in seen:
                past_list.append({"condition": name})
                seen.add(name)
        past_section["medical_history"] = past_list
        new_hist["past_history"] = past_section

    return new_hist


def parse_followup_response(
    followup_question: str,
    followup_answer: str,
    questions: list[dict],
    state: MedicalState,
) -> dict:
    """LLM 翻译用户回答 → 返回 state 增量更新 dict(merge 到 current state)。

    抽出来给 ⑦ process_followup_answer **和** intake_followup_ask multi-interrupt 收完
    后共用 —— 两者都是"把用户自由文本回答结构化",LLM 调用契约一致(中安全等级,
    失败抛异常),只是触发位置不同。

    返回 dict 不含 followup_round / followup_question / followup_questions 字段;
    调用方按自己语义决定要不要更新这些(⑦ 写 round+=1 + clear question/questions;
    intake 在 multi-interrupt 收完后调,后续 ⑤ 会重新决定 questions)。
    """
    prompt = build_followup_parse_prompt(
        followup_question=followup_question,
        followup_answer=followup_answer,
        questions=questions,
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
    already_known = set(confirmed) | set(denied) | set(uncertain)
    for term in result.new_symptoms:
        if term and term not in already_known:
            confirmed.append(term)
            already_known.add(term)

    new_slots = _apply_slot_fills(state.present_illness_slots, result.slot_fills)
    new_medical_history = _merge_obstetric_fills(state.medical_history, result.obstetric_fills)
    new_medical_history = _merge_history_fills(new_medical_history, result.history_fills)

    # present_illness 追加新维度信息:自然语言句式拼接,避免机器格式 k=v 拉低下轮
    # build_query LLM 改写 dense_query 的质量
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
        "medical_history": new_medical_history,
    }


def process_followup_answer(state: MedicalState) -> dict:
    """⑤/⑥ 之后:LLM 翻译追问回答 + followup_round +=1 + clear 当轮 question/questions。

    与 intake 末尾 multi-interrupt 之后调 parse_followup_response 的区别:
    本节点是"⑥ interrupt 后跑了一轮 form",所以 round +=1;并清掉当轮 questions 让
    下游 ⑤ 看到空 → 重新 LLM 判 targeted(或退出)。
    """
    update = parse_followup_response(
        followup_question=state.followup_question,
        followup_answer=state.followup_answer,
        questions=state.followup_questions,
        state=state,
    )
    update["followup_round"] = state.followup_round + 1
    update["followup_question"] = ""
    # 清空 followup_questions —— ⑤ 看到空 → 走 LLM 决定要不要继续追问(intake 阶段)
    # 或 → router 看 source=diagnostic 走 ②(④-driven 鉴别诊断追问)
    update["followup_questions"] = []
    return update
