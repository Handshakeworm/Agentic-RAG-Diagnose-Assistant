"""src/agent/nodes/generate_followup.py — Agent ⑤ generate_followup(DEV_SPEC §4.1.2 ⑤+⑥)。

**双职责**:
1. 已经有 followup_questions(④ 鉴别诊断追问写入)→ LLM 把 questions 合并成 2-3 句
   口语化追问写入 followup_question(短路:若 followup_question 已被写直接透传)
2. followup_questions 为空 + `followup_source=="intake"`(intake 末尾设;⑦ 后回 ⑤
   时是空) → **LLM holistic 看 state 决定 targeted 追问**:
   - 觉得患者答得不好 / 还有要问的 → 输出 targeted 题列表 + open 兜底,写入
     followup_questions + followup_question(后续走 ⑥/⑦)
   - 觉得够了 → 写 followup_questions=[](路由层看到空 → 出口走 ②)

拆分点:LLM 调用与 interrupt 等待分属两个节点(⑤ / ⑥),避免 interrupt 恢复时
重复调 LLM(spec §4.1.2 ⑤+⑥ 拆分设计注)。

LLM 调用契约(§9.1):
  - 文案合并(`free_text` schema 标签):中安全等级,失败抛
  - targeted 判定(`TargetedFollowupOutput`):中安全,失败兜底 = 直接空 questions
    (出口走 ②,信任 ⑩ 诊断的失败兜底,不阻塞流水线)
"""
from __future__ import annotations

import logging
import time

from config.settings import settings
from src.agent.schemas.intake import TargetedFollowupOutput
from src.agent.state import MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_followup_question_prompt, build_targeted_followup_prompt


_logger = logging.getLogger(__name__)
_NODE = "generate_followup"
_SCHEMA = "free_text"
_TARGETED_NODE = "generate_followup_targeted"
_TARGETED_SCHEMA = "TargetedFollowupOutput"


# ────────────────────────────────────────────────────────────────────────────
# 辅助:state.medical_history 摘要 + filled_slots 提取(给 targeted prompt 用)
# ────────────────────────────────────────────────────────────────────────────


def _filled_slots(slots) -> dict:
    return {k: v for k, v in slots.model_dump().items() if v}


def _summarize_history(history: dict) -> str:
    if not history:
        return "  (档案为空)"
    lines: list[str] = []
    bi = history.get("basic_info") or {}
    if bi.get("gender"):
        lines.append(f"  - 性别:{bi['gender']}")
    if bi.get("birth_date"):
        lines.append(f"  - 出生日期:{bi['birth_date']}")
    al = history.get("allergy_history") or []
    al_names = ", ".join(r.get("substance", "?") for r in al) if al else "(未询问)"
    lines.append(f"  - 过敏:{al_names}")
    med = history.get("medication_history") or []
    med_names = ", ".join(r.get("drug_name", "?") for r in med) if med else "(未询问)"
    lines.append(f"  - 长期/在用药:{med_names}")
    past = (history.get("past_history") or {}).get("medical_history") or []
    past_names = ", ".join(r.get("condition", "?") for r in past) if past else "(未询问)"
    lines.append(f"  - 既往疾病:{past_names}")
    ob = history.get("obstetric_history")
    if ob:
        lines.append(
            f"  - 妊娠/哺乳:{ob.get('pregnancy_status', '?')} / {ob.get('lactation_status', '?')}"
        )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# intake targeted phase:LLM 看 holistic 决定 slot 之外的追问
# ────────────────────────────────────────────────────────────────────────────


def _decide_intake_targeted(state: MedicalState) -> list[dict]:
    """LLM 看 filled_slots holistic 决定 targeted 追问;失败兜底返空。

    返回 list of dict(每条带 type=targeted,question 已填);空 list 表示"够了"。
    """
    prompt = build_targeted_followup_prompt(
        chief_complaint=state.chief_complaint,
        present_illness=state.present_illness,
        filled_slots=_filled_slots(state.present_illness_slots),
        confirmed_symptoms=list(state.confirmed_symptoms),
        denied_symptoms=list(state.denied_symptoms),
        uncertain_symptoms=list(state.uncertain_symptoms),
        medical_history_summary=_summarize_history(state.medical_history),
        quota=settings.agent_limits.MAX_FOLLOWUP_QUESTIONS,
    )

    _attempts.labels(node=_TARGETED_NODE, schema=_TARGETED_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(
            TargetedFollowupOutput, method="json_mode"
        ).with_retry(stop_after_attempt=3)
        result: TargetedFollowupOutput = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _TARGETED_NODE, "schema": _TARGETED_SCHEMA},
            },
        )
    except Exception as e:
        _failures.labels(
            node=_TARGETED_NODE, schema=_TARGETED_SCHEMA, exception_type=type(e).__name__,
        ).inc()
        _logger.warning(
            "[%s] targeted decide LLM failed, fallback to empty (let router exit): %s",
            _TARGETED_NODE, e,
        )
        return []  # 中安全:失败 → 空(出口走 ②),不阻塞流水线
    finally:
        _latency.labels(node=_TARGETED_NODE, schema=_TARGETED_SCHEMA).observe(
            time.perf_counter() - t0
        )

    targeted: list[dict] = [
        {"type": "targeted", "question": item.question, "target": item.target}
        for item in result.questions[: settings.agent_limits.MAX_FOLLOWUP_QUESTIONS - 1]
    ]
    if targeted:
        targeted.append({"type": "open"})  # 每批 form 都让患者有机会补充症状
    return targeted


# ────────────────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────────────────


def generate_followup(state: MedicalState) -> dict:
    """生成自然语言追问问题。两种触发场景:

    a) `followup_questions` 已写(④ 鉴别诊断追问 / 上轮残留)→ 拼文案
    b) `followup_questions` 空 + `followup_source==intake` → LLM 判 targeted

    出口:由 `generate_followup_router` 看 followup_questions 决定 ⑥ or ②
    """
    # ── 场景 b:intake 阶段 ⑦ loop 回来,LLM 判 targeted ──
    if not state.followup_questions and state.followup_source == "intake":
        targeted = _decide_intake_targeted(state)
        if not targeted:
            # LLM 觉得够了 → 空 questions,出口走 ② 检索
            return {"followup_questions": [], "followup_question": ""}
        text = _build_targeted_text(targeted)
        return {"followup_questions": targeted, "followup_question": text}

    # ── 场景 a:followup_questions 已存在(④ 写的 / intake 阶段刚拼好的)──
    if state.followup_question:
        # 上游已模板拼好(intake 内部弹 form 时已写,这里防御短路)
        _logger.debug("[%s] followup_question 已存在,跳过 LLM 调用", _NODE)
        return {}

    if not state.followup_questions:
        # 没题且非 intake 阶段(④ 阶段不该到这,防御兜底)
        return {"followup_question": ""}

    prompt = build_followup_question_prompt(
        chief_complaint=state.chief_complaint,
        questions=state.followup_questions,
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        llm = get_llm()
        chain = llm.with_retry(stop_after_attempt=3)
        msg = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA},
            },
        )
        question = (
            msg.content if hasattr(msg, "content") else str(msg)
        ).strip()
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _logger.error("[%s] free-text LLM call failed: %s", _NODE, e)
        raise
    finally:
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(
            time.perf_counter() - t0
        )

    return {"followup_question": question}


def _build_targeted_text(questions: list[dict]) -> str:
    """targeted+open 列表 → 用户可见文案;target=open 走模板"还有别的不舒服…"。"""
    _OPEN_TEMPLATE = "您还有没有别的地方不舒服?有就一起说,我好综合判断。"
    parts: list[str] = []
    for q in questions:
        if q.get("type") == "open":
            parts.append(_OPEN_TEMPLATE)
        else:
            text = q.get("question") or q.get("text") or ""
            if text:
                parts.append(text)
    return "\n\n".join(parts)
