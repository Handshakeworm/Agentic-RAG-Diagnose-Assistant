"""src/agent/nodes/generate_followup.py — Agent ⑤ generate_followup(DEV_SPEC §4.1.2 ⑤+⑥)。

**2 步 LLM,双 list 出口,检索前 holistic gate**(2026-05-21 拆分 / 2026-05-22 Step A 改 flash)。

定位:进 ②③④ 检索/鉴别前的最后一道 holistic gate。intake 把 13 维 slot 灌完后,
⑤ 用 **2 次 LLM** 决策"还差什么信息",按"患者能不能答"分两路:
  - Step A(flash):holistic 看全量 state 出 `askable_targets`(英文短标签)
    + `unaskable_findings`(医生侧 description + reason)
  - Step B(flash,仅 askable_targets 非空才调):把短标签 → 患者侧自然中文问句

2026-05-22 改造:Step A 从 pro reasoner 改用 flash。原因:pro reasoner thinking
45-180s 不稳定,且 reasoner + json_mode 嵌套 schema 偶发字段名飘 → with_retry ×
60s wait_exponential 撞 retry storm 卡 3+ 分钟。任务本质是"对照 state 找信息缺口",
不需 reasoner 深度;flash 5-15s 完成,嵌套 schema 输出稳定(同款经验:info_collect)。

出口优先级(节点内决定 + router 据此分流):
  askable_questions 非空 → `to_wait` → ⑥
  否则 unaskable_findings 非空 → `to_recommend_exam` → ⑧a(首诊模式)
  都空 → `to_build_query` → ②

**每次都把 `unaskable_findings` 写到 `state.unaskable_symptoms`**(覆盖上轮),
保证下游(⑩ 或 ⑧a)始终能消费 ⑤ 累积的 unaskable 列表。

LLM 调用 2 处(中安全等级):
  - Step A `HolisticGateDecision`:`settings.llm.FAST_MODEL_NAME`(flash);失败 → 两 list 都空,router 走 ②
  - Step B `QuestionGenOutput`:`settings.llm.FAST_MODEL_NAME`(flash);失败 → fallback
    用 target 短标签当 question 文本(降级体验,不阻塞,因为 Step A 决策已成功)
"""
from __future__ import annotations

import logging
import time

from langchain_core.callbacks import dispatch_custom_event

from config.settings import settings
from src.agent.schemas.intake import (
    HolisticGateDecision,
    QuestionGenOutput,
    TargetedFollowupItem,
)
from src.agent.state import MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import (
    build_question_generation_prompt,
    build_targeted_followup_prompt,
)


_logger = logging.getLogger(__name__)
_NODE = "generate_followup"
_SCHEMA_STEP_A = "HolisticGateDecision"
_SCHEMA_STEP_B = "QuestionGenOutput"


# ────────────────────────────────────────────────────────────────────────────
# 辅助:state.medical_history 摘要 + filled_slots 提取(给 prompt 用)
# ────────────────────────────────────────────────────────────────────────────


def _filled_slots(slots) -> dict:
    return {k: v for k, v in slots.model_dump().items() if v}


def _summarize_history(history: dict) -> str:
    """档案 → 病史摘要文本。空 list 显示"已询问,无"(⓪a 入站必问过),不是"未询问",
    防止 ⑤ LLM 把已问无的范畴再次列进 askable 重复追问。"""
    if not history:
        return "  (档案为空)"
    lines: list[str] = []
    bi = history.get("basic_info") or {}
    if bi.get("gender"):
        lines.append(f"  - 性别:{bi['gender']}")
    if bi.get("birth_date"):
        lines.append(f"  - 出生日期:{bi['birth_date']}")
    al = history.get("allergy_history") or []
    al_names = ", ".join(r.get("substance", "?") for r in al) if al else "已询问,无"
    lines.append(f"  - 过敏:{al_names}")
    med = history.get("medication_history") or []
    med_names = ", ".join(r.get("drug_name", "?") for r in med) if med else "已询问,无"
    lines.append(f"  - 长期/在用药:{med_names}")
    past = (history.get("past_history") or {}).get("medical_history") or []
    past_names = ", ".join(r.get("condition", "?") for r in past) if past else "已询问,无"
    lines.append(f"  - 既往疾病:{past_names}")
    ob = history.get("obstetric_history")
    if ob:
        lines.append(
            f"  - 妊娠/哺乳:{ob.get('pregnancy_status', '?')} / {ob.get('lactation_status', '?')}"
        )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Step A:flash holistic 决策(出 askable_targets + unaskable_findings)
# 中安全等级,失败兜底返"两 list 都空"(让 router 走 ② 检索,不阻塞流水线)
#
# 2026-05-22 改用 flash:之前用 pro reasoner,thinking 时间 45-180s 不稳定,
# 且 reasoner + json_mode 嵌套 schema(unaskable_findings: list[{description, reason}])
# 偶发字段名飘 → with_retry × 60s wait_exponential 撞 retry storm 卡 3+ 分钟。
# flash 非 thinking,5-15s 完成,嵌套 schema 输出稳定(参考 info_collect 同款经验)。
# trade-off:决策推理深度下降,但本节点任务是"对照已知 state 找信息缺口",
# 不需要 reasoner 级深度,flash 经过验证决策质量过关。
# ────────────────────────────────────────────────────────────────────────────


def _llm_holistic_gate_flash(state: MedicalState) -> HolisticGateDecision:
    """Step A:flash 看全量 state 一次判 askable_targets + unaskable;
    失败兜底返空(router 走 ②)。"""
    # DEBUG remove: 临时观察 ⑤ Step A 实际看到的 state 三类 + slots,验证 multi-round 稳定性
    _logger.info(
        "[debug] ⑤ stepA_input: confirmed=%s denied=%s uncertain=%s slots=%s",
        list(state.confirmed_symptoms),
        list(state.denied_symptoms),
        list(state.uncertain_symptoms),
        _filled_slots(state.present_illness_slots),
    )
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

    _attempts.labels(node=_NODE, schema=_SCHEMA_STEP_A).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm(model=settings.llm.FAST_MODEL_NAME).with_structured_output(
            HolisticGateDecision, method="json_mode"
        ).with_retry(stop_after_attempt=3)
        result: HolisticGateDecision = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA_STEP_A},
            },
        )
        # DEBUG remove: 看 Step A 决策出口,定位重复问题是 Step A 选错还是 Step B 拼错
        _logger.info(
            "[debug] ⑤ stepA_output: askable_targets=%s unaskable_findings=%s",
            list(result.askable_targets),
            [{"desc": u.description, "reason": u.reason} for u in result.unaskable_findings],
        )
        return result
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA_STEP_A, exception_type=type(e).__name__,
        ).inc()
        _logger.warning(
            "[%s] Step A holistic gate failed, fallback to empty lists (router → ②): %s",
            _NODE, e,
        )
        return HolisticGateDecision()  # 两 list 都默认空
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA_STEP_A).observe(elapsed)
        _logger.info("[%s] Step A flash decide elapsed=%.2fs", _NODE, elapsed)


# ────────────────────────────────────────────────────────────────────────────
# Step B:flash 把 askable_targets 拼成自然中文问句
# 中安全等级,失败 fallback 用 target 短标签当 question(降级体验,不阻塞)
# ────────────────────────────────────────────────────────────────────────────


def _llm_question_gen_flash(
    state: MedicalState, askable_targets: list[str]
) -> list[TargetedFollowupItem]:
    """Step B:flash 把每个 target 转成自然中文问句;失败 fallback 用短标签拼。"""
    prompt = build_question_generation_prompt(
        chief_complaint=state.chief_complaint,
        present_illness=state.present_illness,
        askable_targets=askable_targets,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA_STEP_B).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm(model=settings.llm.FAST_MODEL_NAME).with_structured_output(
            QuestionGenOutput, method="json_mode"
        ).with_retry(stop_after_attempt=3)
        result: QuestionGenOutput = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA_STEP_B},
            },
        )
        # DEBUG remove: 看 Step B 拼出来的中文问句,对比 askable_targets 看是否拼歪
        _logger.info(
            "[debug] ⑤ stepB_output: questions=%s",
            [{"target": q.target, "question": q.question} for q in result.questions],
        )
        return list(result.questions)
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA_STEP_B, exception_type=type(e).__name__,
        ).inc()
        _logger.warning(
            "[%s] Step B question gen failed, fallback to short labels: %s",
            _NODE, e,
        )
        # 降级:用短标签拼朴素问句,不阻塞流程
        return [
            TargetedFollowupItem(
                question=f"请补充一下:{t.replace('_', ' ')}",
                target=t,
            )
            for t in askable_targets
        ]
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA_STEP_B).observe(elapsed)
        _logger.info("[%s] Step B flash qgen elapsed=%.2fs", _NODE, elapsed)


# ────────────────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────────────────


def generate_followup(state: MedicalState) -> dict:
    """检索前 holistic gate:Step A pro 决策 + Step B flash 拼问句,按优先级分流。

    出口(由 `generate_followup_out_router` 看 state 决定):
      - askable 非空 → ⑥ wait_followup_answer
      - 无 askable 但有 unaskable → ⑧a recommend_exam(首诊模式)
      - 都空 → ② build_query
    """
    # Step A:flash 决策(灰字"正在准备需要确认的细节…"由节点 chain_start 默认触发)
    decision = _llm_holistic_gate_flash(state)

    quota = settings.agent_limits.MAX_FOLLOWUP_QUESTIONS
    # 留 1 个位置给 open 兜底,所以 askable 最多 quota - 1 条
    askable_targets = list(decision.askable_targets)[: quota - 1]
    unaskable_items = list(decision.unaskable_findings)[:quota]

    # 每次都把 unaskable 写回 state(覆盖上轮),供下游 ⑩/⑧a 消费
    unaskable_dump: list[dict] = [u.model_dump() for u in unaskable_items]

    if askable_targets:
        # Step B 前推 custom event,让前端把灰字切到"正在生成追问…"
        # (Step A/B 同节点,langgraph chain_start 只 fire 一次,要靠 custom event 加段)
        dispatch_custom_event("progress_step", {"step": "step_b_question_gen"})
        # Step B:flash 把短标签 → 自然中文问句
        askable_items = _llm_question_gen_flash(state, askable_targets)
        questions: list[dict] = [
            {"type": "targeted", "question": q.question, "target": q.target}
            for q in askable_items
        ]
        # 每批 form 留 open 兜底捕获遗漏症状;question 字段必填(前端按它渲染),
        # 否则 UI 显示"(无问题文本)"
        questions.append({"type": "open", "question": _OPEN_TEMPLATE})
        text = _build_text(questions)
        return {
            "followup_questions": questions,
            "followup_question": text,
            "unaskable_symptoms": unaskable_dump,
        }

    if unaskable_items:
        # 无 askable 但有 unaskable:走 ⑧a 首诊推单(不写 followup_questions,router 据此判)
        return {
            "followup_questions": [],
            "followup_question": "",
            "unaskable_symptoms": unaskable_dump,
        }

    # 都空:信息已足 → ②
    return {
        "followup_questions": [],
        "followup_question": "",
        "unaskable_symptoms": unaskable_dump,  # 仍写(可能是 [] 覆盖上轮残留)
    }


_OPEN_TEMPLATE = "您还有没有别的地方不舒服?有就一起说,我好综合判断。"


def _build_text(questions: list[dict]) -> str:
    """targeted+open 列表 → 用户可见文案;type=open 走模板"还有别的不舒服…"。"""
    parts: list[str] = []
    for q in questions:
        if q.get("type") == "open":
            parts.append(_OPEN_TEMPLATE)
        else:
            text = q.get("question") or q.get("text") or ""
            if text:
                parts.append(text)
    return "\n\n".join(parts)
