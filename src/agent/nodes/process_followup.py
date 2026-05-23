"""src/agent/nodes/process_followup.py — Agent ⑦ process_followup_answer(DEV_SPEC §4.1.2 ⑦)。

LLM 解析患者回答 → 维度级槽位回填 + 新症状提取。followup_round += 1 后回到
build_query 复跑流水线。

④ 已重设计为只产 slot / open 两类追问,⑦ 不再有"症状级 yes/no 回答分流"分支 —
开放式追问得到的新症状统一进 `new_symptoms` 字段,本节点直接 append 到 confirmed_symptoms。

**模型选择**:用 `settings.llm.FAST_MODEL_NAME`(deepseek-v4-flash),不走主链路 pro reasoner。
任务性质是"把患者自由文本归类到 slot/symptom/history",纯文本结构化,不需要 reasoner
推理深度;且 pro reasoner 对本 schema(FollowupParseResult 含 union value + Optional dict)
在 thinking 阶段会自由发挥字段名,导致 LangChain with_retry 反复重试,单次拖到 2 分钟。
flash 非 thinking,1-3 秒结束;参考 info_collect 用 flash 跑同类任务一直稳。

中安全等级:失败 → 抛异常终止会话(回答未解析将导致信息丢失,不能静默)。
"""
from __future__ import annotations

import logging
import re
import time

from config.settings import settings
from src.agent.schemas.followup import FollowupParseResult
from src.agent.state import SLOT_UNKNOWN_SENTINEL, MedicalState, PresentIllnessSlots
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_followup_parse_prompt


_logger = logging.getLogger(__name__)
_NODE = "process_followup_answer"
_SCHEMA = "FollowupParseResult"

_MULTI_VALUE_SLOTS = {
    "aggravating", "relieving", "associated_symptoms",
    "trigger", "nature", "severity",  # 2026-05-22:str → list[str] 解决多值覆盖丢失
    "treatments",  # 2026-05-22:treatment_tried+treatment_response 合并为 treatments
}

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
    "treatments": "诊疗经过:{value}",  # value 是 list,每条 '<治疗>: <反应>' 顿号串拼起来
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


_FAMILY_ENTRY_RE = re.compile(
    r"^\s*(?P<relation>[^:：]+?)\s*[:：]\s*(?P<condition>.+?)"
    r"(?:\s*[(（]\s*(?:发病)?\s*(?P<onset_age>\d+)\s*岁?\s*[)）])?\s*$"
)


def _parse_history_entry(raw: str) -> tuple[str, str | None]:
    """切 'X:Y' 半结构化字符串。返回 (左, 右|None)。失败 → ("", None)。

    覆盖全角/半角冒号;LLM 漂格式时尽量保留;失败兜底由调用方处理(保留原文 notes)。
    """
    if not raw or not isinstance(raw, str):
        return "", None
    for sep in (":", ":"):
        if sep in raw:
            left, _, right = raw.partition(sep)
            return left.strip(), (right.strip() or None)
    return raw.strip(), None


def _merge_history_fills(
    medical_history: dict, fills: dict[str, list[str]] | None
) -> dict:
    """把 history_fills 5 段 list 追加到 medical_history 对应位置。

    字段映射对齐 `patient_repo.load_medical_history` 返回结构 + 2026-05-22 新增 2 段:
      - allergies        → allergy_history[].substance
      - medications      → medication_history[].drug_name(is_current=True)
      - past_conditions  → past_history.medical_history[].condition
      - family_conditions(新):
          '<关系>:<疾病>[(发病<年龄>岁)]' 阳性 → family_history[].{relation, condition, onset_age, notes}
          '<疾病>:无' 已问无       → family_history_asked_no[](疾病名 dedup 追加)
      - personal_conditions(新):
          '<项目>:<详细>' 有内容    → personal_history.dynamic_notes[](原文 dedup 追加)
          '<项目>:无' 已问无        → personal_history_asked_no[](项目名 dedup 追加)

    解析失败兜底:阳性 entry 失败 → 整段塞 notes 字段(沉淀节点未来再处理),不丢数据。
    幂等:同名条目跳过。fills 为 None → 透传;某段缺省 = 本轮无史类问询。
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

    # 家族史 — 阳性 entry 入 family_history,'<疾病>:无' 入 family_history_asked_no
    family_new = fills.get("family_conditions") or []
    if family_new:
        family_list = list(new_hist.get("family_history") or [])
        asked_no_list = list(new_hist.get("family_history_asked_no") or [])
        seen_positive = {  # (relation, condition) 双键去重
            (r.get("relation"), r.get("condition"))
            for r in family_list if r.get("relation") and r.get("condition")
        }
        seen_asked_no = set(asked_no_list)
        for raw in family_new:
            left, right = _parse_history_entry(raw)
            if not left:
                continue  # 整段空白,跳过
            # 形如 "胆结石:无" → 已问无的疾病范畴
            if right and right.strip() in ("无", "没有", "否"):
                if left not in seen_asked_no:
                    asked_no_list.append(left)
                    seen_asked_no.add(left)
                continue
            # 形如 "父亲:糖尿病(发病50岁)" → 阳性 entry
            m = _FAMILY_ENTRY_RE.match(raw)
            if m:
                relation = m.group("relation").strip()
                condition = m.group("condition").strip()
                onset_age = int(m.group("onset_age")) if m.group("onset_age") else None
                key = (relation, condition)
                if key not in seen_positive:
                    family_list.append({
                        "relation": relation,
                        "condition": condition,
                        "onset_age": onset_age,
                        "notes": raw,  # 保留原文给沉淀节点
                    })
                    seen_positive.add(key)
            else:
                # 解析失败 — 保留整段到 notes(沉淀节点未来用 LLM 再解析)
                family_list.append({
                    "relation": None,
                    "condition": None,
                    "onset_age": None,
                    "notes": raw,
                })
        new_hist["family_history"] = family_list
        new_hist["family_history_asked_no"] = asked_no_list

    # 个人史 — 有内容入 dynamic_notes,'<项目>:无' 入 personal_history_asked_no
    personal_new = fills.get("personal_conditions") or []
    if personal_new:
        personal_dict = dict(new_hist.get("personal_history") or {})
        dynamic_notes = list(personal_dict.get("dynamic_notes") or [])
        asked_no_list = list(new_hist.get("personal_history_asked_no") or [])
        seen_notes = set(dynamic_notes)
        seen_asked_no = set(asked_no_list)
        for raw in personal_new:
            if not raw or not isinstance(raw, str):
                continue
            left, right = _parse_history_entry(raw)
            if right and right.strip() in ("无", "没有", "否"):
                if left and left not in seen_asked_no:
                    asked_no_list.append(left)
                    seen_asked_no.add(left)
            else:
                # 有内容(含解析失败时整段当 note)→ append 原文
                if raw not in seen_notes:
                    dynamic_notes.append(raw)
                    seen_notes.add(raw)
        personal_dict["dynamic_notes"] = dynamic_notes
        new_hist["personal_history"] = personal_dict
        new_hist["personal_history_asked_no"] = asked_no_list

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
        chain = get_llm(model=settings.llm.FAST_MODEL_NAME).with_structured_output(
            FollowupParseResult, method="json_mode"
        ).with_retry(stop_after_attempt=3)
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
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(elapsed)
        _logger.info("[%s] parse_followup_response flash elapsed=%.2fs", _NODE, elapsed)

    confirmed = list(state.confirmed_symptoms)
    denied = list(state.denied_symptoms)
    uncertain = list(state.uncertain_symptoms)

    # 症状三分类合并:LLM 按语气把患者提及的症状分到 confirmed/denied/uncertain。
    # intake 路径不经过 ② build_query,如果不在此处写,⑤ 第一次进来时 denied/uncertain
    # 永远是空,会反复问"呕吐没?"/"头晕没?"已经回答过的症状。
    already_known = set(confirmed) | set(denied) | set(uncertain)
    for term in result.confirmed_symptoms:
        if term and term not in already_known:
            confirmed.append(term)
            already_known.add(term)
    for term in result.denied_symptoms:
        if term and term not in already_known:
            denied.append(term)
            already_known.add(term)
    for term in result.uncertain_symptoms:
        if term and term not in already_known:
            uncertain.append(term)
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

    # DEBUG remove: 临时观察 ⑦ flash 翻译实际写到三类的内容,验证 multi-round denied 累积稳定性
    _logger.info(
        "[debug] ⑦ parse_result: llm_confirmed=%s llm_denied=%s llm_uncertain=%s "
        "merged_confirmed=%s merged_denied=%s merged_uncertain=%s slot_fills=%s",
        list(result.confirmed_symptoms), list(result.denied_symptoms), list(result.uncertain_symptoms),
        confirmed, denied, uncertain, result.slot_fills,
    )
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
