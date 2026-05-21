"""src/agent/nodes/select_symptom.py — Agent ④ select_discriminative_symptom(DEV_SPEC §4.1.2 ④)。

1 次 LLM 调用直接选追问项,不再走"TF-IDF 抽症状 + 信息增益 + 可问性评估"那条
重工程化路径(实测 ④ TF-IDF 抽出来 94% 是医学教材通用高频词而非鉴别症状,
信息增益的统计基础不成立 — 见 EL_DESIGN_REVIEW §11)。

LLM 任务:
  - 从 13 维 HPI 空缺槽里挑 1~2 个对当前主诉**诊断价值最高**的维度问
    (优先 patient-answerable 维度:时间/部位/性质/诱因/缓解等)
  - 如果空缺维度都不重要 / 已大部分填满,加一条 open 式 "还有别的不舒服吗?"
    作为兜底捕获遗漏症状
  - 总数 ≤ MAX_FOLLOWUP_QUESTIONS

LLM 调用 1 处,按 §9.1 中安全级模板独立写 try/except/finally 埋点。
"""
from __future__ import annotations

import logging
import time

from config.settings import settings
from src.agent.nodes.intake_followup_ask import (
    _attach_question_texts,
    _build_question_text,
)
from src.agent.schemas.symptom_selection import SmartFollowupOutput
from src.agent.state import MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_smart_followup_prompt


_logger = logging.getLogger(__name__)


def _empty_slots(slots) -> list[str]:
    """从 PresentIllnessSlots 抽空槽列表(单值=None,多值=[])。"""
    out: list[str] = []
    data = slots.model_dump()
    for k, v in data.items():
        if v is None or v == []:
            out.append(k)
    return out


def _filled_slots(slots) -> dict:
    """已填槽 dict(供 prompt 展示)。"""
    return {k: v for k, v in slots.model_dump().items() if v}


# ────────────────────────────────────────────────────────────────────────────
# LLM 调用 — Smart followup(中安全,失败 → 空 followup 直接跳诊断)
# ────────────────────────────────────────────────────────────────────────────


def _call_smart_followup(state: MedicalState) -> SmartFollowupOutput:
    node, schema = "select_symptom_smart_followup", "SmartFollowupOutput"
    _attempts.labels(node=node, schema=schema).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(
            SmartFollowupOutput, method="json_mode"
        ).with_retry(stop_after_attempt=3)
        prompt = build_smart_followup_prompt(
            chief_complaint=state.chief_complaint,
            present_illness=state.present_illness,
            filled_slots=_filled_slots(state.present_illness_slots),
            empty_slots=_empty_slots(state.present_illness_slots),
            confirmed_symptoms=list(state.confirmed_symptoms),
            denied_symptoms=list(state.denied_symptoms),
            uncertain_symptoms=list(state.uncertain_symptoms),
            quota=settings.agent_limits.MAX_FOLLOWUP_QUESTIONS,
        )
        return chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": node, "schema": schema},
            },
        )
    except Exception as e:
        _failures.labels(
            node=node, schema=schema, exception_type=type(e).__name__
        ).inc()
        _logger.warning(
            "[%s] smart followup failed, fall back to empty followup: %s", node, e,
        )
        # spec §9.3 中安全等级失败处理:跳过本轮追问,直接进诊断
        return SmartFollowupOutput(questions=[])
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=node, schema=schema).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", node, elapsed)


# ────────────────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────────────────


def _needs_obstetric_force_ask(state: MedicalState) -> bool:
    """首诊女性 + 档案无 obstetric → 强制问妊娠/哺乳。

    Why: ⑪ safety_gate 规则层硬依赖 `obstetric_history.pregnancy_status` /
    `lactation_status` 才能触发妊娠禁忌兜底。生产里 obstetric 子表的写入路径
    要么靠 App 端 PUT /me/obstetric 主动填,要么靠这里强制追问 — 不问就永远 None。

    只在 followup_round == 0(首诊轮)强制问一次,避免患者回答不明时反复追问。
    """
    if state.followup_round != 0:
        return False
    history = state.medical_history or {}
    basic = history.get("basic_info") or {}
    if basic.get("gender") != "female":
        return False
    return history.get("obstetric_history") is None


def select_discriminative_symptom(state: MedicalState) -> dict:
    """1 次 LLM 同时出 3 件事:追问项 + unaskable 粗筛 + obstetric 强制问。

    - `followup_questions`:追问项(slot 维度补全 / open 开放式 / obstetric 妊娠哺乳);空 → 路由跳诊断
    - `unaskable_symptoms`:LLM 想知道但患者答不上的体征/指标(粗筛版),
      ⑩ 会基于诊断结果再次校准并覆盖此字段
    - 首诊女性 + obstetric 档案空 → **强制 prepend 1 条 obstetric 追问**,
      让 ⑪ safety_gate 妊娠禁忌兜底能在本会话内生效
    """
    result = _call_smart_followup(state)
    # 边界校验:slot type 必须带 slot 名;open/obstetric 不带 slot
    followup_questions: list[dict] = []
    for q in result.questions:
        if q.type == "slot":
            if not q.slot:
                continue  # 缺 slot 名的 slot type 丢弃
            followup_questions.append({"type": "slot", "slot": q.slot})
        elif q.type == "open":
            followup_questions.append({"type": "open"})
        # LLM 不应该自己产 obstetric type — 那是节点入口强制 prepend 的,见下
    # 强制 obstetric 追问 prepend 到最前(诊前安全兜底优先级最高)
    if _needs_obstetric_force_ask(state):
        followup_questions.insert(0, {"type": "obstetric"})
    # 截到 quota(LLM 已经约束 max_length=5 但兜底再截;obstetric 必在,挤掉末尾 LLM 项)
    followup_questions = followup_questions[: settings.agent_limits.MAX_FOLLOWUP_QUESTIONS]
    # unaskable 粗筛同样兜底截到 quota
    unaskable: list[dict] = [u.model_dump() for u in result.unaskable_symptoms]
    unaskable = unaskable[: settings.agent_limits.MAX_FOLLOWUP_QUESTIONS]
    # ④ → ⑥ 直连(跳过 ⑤):本节点产题方负责同时拼好 followup_question 文案,
    # ⑥ 入口零拼装直接 interrupt(state.followup_question)。空 list → 空文案,路由跳诊断。
    if followup_questions:
        followup_questions = _attach_question_texts(followup_questions)
        followup_question_text = _build_question_text(followup_questions)
    else:
        followup_question_text = ""
    return {
        "followup_questions": followup_questions,
        "followup_question": followup_question_text,
        "unaskable_symptoms": unaskable,
    }
