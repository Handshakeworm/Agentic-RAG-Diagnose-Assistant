"""src/agent/nodes/select_symptom.py — Agent ⑤ select_discriminative_symptom(DEV_SPEC §4.1.2 ⑤)。

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
        _latency.labels(node=node, schema=schema).observe(
            time.perf_counter() - t0
        )


# ────────────────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────────────────


def select_discriminative_symptom(state: MedicalState) -> dict:
    """1 次 LLM 同时出 3 件事:追问项 + unaskable 粗筛 + 信息增益占位。

    - `followup_questions`:追问项(slot 维度补全 / open 开放式);空 → 路由跳诊断
    - `unaskable_symptoms`:LLM 想知道但患者答不上的体征/指标(粗筛版),
      ⑩ Step 3 会基于诊断结果再次校准并覆盖此字段
    """
    result = _call_smart_followup(state)
    # 边界校验:slot type 必须带 slot 名;open type 不带
    followup_questions: list[dict] = []
    for q in result.questions:
        if q.type == "slot":
            if not q.slot:
                continue  # 缺 slot 名的 slot type 丢弃
            followup_questions.append({"type": "slot", "slot": q.slot})
        elif q.type == "open":
            followup_questions.append({"type": "open"})
    # 截到 quota(LLM 已经约束 max_length=5 但兜底再截)
    followup_questions = followup_questions[: settings.agent_limits.MAX_FOLLOWUP_QUESTIONS]
    # unaskable 粗筛同样兜底截到 quota
    unaskable: list[dict] = [u.model_dump() for u in result.unaskable_symptoms]
    unaskable = unaskable[: settings.agent_limits.MAX_FOLLOWUP_QUESTIONS]
    return {
        "followup_questions": followup_questions,
        "unaskable_symptoms": unaskable,
    }
