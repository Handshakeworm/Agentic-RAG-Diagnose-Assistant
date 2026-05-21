"""src/agent/nodes/intake_followup_ask.py — 入站 slot 收集 + 翻译节点(DEV_SPEC §4.1.3)。

**单一职责**:把 PresentIllnessSlots 13 维 slot 一次性问全 + LLM holistic 翻译。

- 节点内 multi-interrupt(分批弹 form,每批 4 slot + 1 open):
  - 第 1 批:onset_time / onset_mode / trigger / location + open
  - 第 2 批:nature / severity / duration_pattern / aggravating + open
  - 第 3 批:relieving / associated_symptoms / progression / treatment_tried + open
  - 第 4 批:treatment_response + open(剩 1 个)
- 全部 batch 收完 → **一次 LLM holistic 翻译**累积的所有 batch answer(复用 ⑦ 的
  `parse_followup_response` helper),把 slot 原文写回 state.present_illness_slots,
  顺带 merge history/obstetric 进 medical_history,append new_symptoms 进 confirmed_symptoms
- 出口固定 → ⑤(由 ⑤ LLM 看 holistic 决定要不要 targeted 追问)。⑦ 后的路由分流不靠
  source 字段,而是看 `candidate_chunks` 是否非空(空 = intake 后还没检索过)。

**约束**:
- 节点只跑一次,不被 loop
- 不直连 ② / ⑦(slot 之外的追问留给 ⑤)
- 多 batch 间 LLM 看到的是所有原文累积(`"\\n\\n".join(...)`),holistic 解析,
  跨 batch 能判一致性 / 矛盾(targeted 反问职责仍归 ⑤,本节点只翻译)

LangGraph 节点内 multi-interrupt:body 重跑时 `interrupt()` 按调用顺序返已 cached
resume value,所以局部变量(accumulated_q/answers/etc.)每次重跑时重新建立 + 累积
到当前 batch — 行为正确。
"""
from __future__ import annotations

from langchain_core.callbacks import dispatch_custom_event
from langgraph.types import interrupt

from src.agent.nodes.process_followup import parse_followup_response
from src.agent.state import MedicalState


# ────────────────────────────────────────────────────────────────────────────
# 模板字符串:零 LLM 拼问题文案,前端 form 渲染按 q.question 字段读
# ────────────────────────────────────────────────────────────────────────────

_SLOT_QUESTION_TEMPLATES: dict[str, str] = {
    "onset_time":         "您是什么时候开始出现这个不适的?",
    "onset_mode":         "症状是突然出现的,还是慢慢出来的?",
    "trigger":            "有没有什么诱因?比如吃了什么、做了什么之后开始的?",
    "location":           "具体是哪个部位不舒服?",
    "nature":             "怎么个不舒服法?(比如刺痛、胀痛、隐痛、烧灼、麻木、酸胀…)",
    "severity":           "程度有多重?有没有影响到睡眠、吃饭或正常活动?",
    "duration_pattern":   "是一直持续,还是一阵阵的?每次持续多久?",
    "aggravating":        "做什么动作或在什么情况下会加重?",
    "relieving":          "做什么能让症状缓解一些?",
    "associated_symptoms":"除了主要的不适,有没有其他同时出现的症状(发热、恶心、出汗等)?",
    "progression":        "和最开始相比,症状是加重、减轻还是没变化?",
    "treatment_tried":    "之前有没有自己吃过药或采取过其他措施?",
    "treatment_response": "用过药的话,效果怎么样?",
}

_TYPE_TEMPLATES: dict[str, str] = {
    "open":      "您还有没有别的地方不舒服?有就一起说,我好综合判断。",
    "history":   "为了用药安全,简单了解一下您的基础情况:有没有食物或药物过敏?有长期服用的药物吗?有高血压、糖尿病这类慢性病吗?(如果没有,直接答'无'即可)",
    "obstetric": "为了用药安全,需要确认一下:您当前是否怀孕?是否在哺乳期?",
}


def _question_text_for(q: dict) -> str:
    """单条 followup_question dict → 用户可见的问题文本(模板查表)。"""
    t = q.get("type")
    if t == "slot":
        return _SLOT_QUESTION_TEMPLATES.get(q.get("slot", ""), "")
    if t == "targeted":
        return q.get("question") or q.get("text") or ""
    return _TYPE_TEMPLATES.get(t or "", "")


def _attach_question_texts(questions: list[dict]) -> list[dict]:
    """副作用:给每条 dict 写入/覆盖 question 字段(前端按 q.question 渲染)。"""
    for q in questions:
        text = _question_text_for(q)
        if text:
            q["question"] = text
    return questions


def _build_question_text(questions: list[dict]) -> str:
    """followup_questions list → 用户可见的问题段落(模板拼,零 LLM)。"""
    parts = [_question_text_for(q) for q in questions]
    return "\n\n".join(p for p in parts if p)


# ────────────────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────────────────


_BATCH_SIZE = 4  # 每批 4 slot 题 + 1 open;13 / 4 ≈ 4 批


def intake_followup_ask(state: MedicalState) -> dict:
    """multi-interrupt 收 13 维 slot batch + LLM holistic 翻译 → 出口 ⑤。"""
    slot_names = list(state.present_illness_slots.model_dump().keys())
    asked: set[str] = set()
    accumulated_questions: list[dict] = []
    accumulated_answers: list[str] = []
    accumulated_texts: list[str] = []

    while True:
        remaining = [s for s in slot_names if s not in asked]
        if not remaining:
            break
        batch_slots = remaining[:_BATCH_SIZE]
        batch_qs: list[dict] = [{"type": "slot", "slot": s} for s in batch_slots]
        batch_qs.append({"type": "open"})  # 每批 open 兜底,让患者随时补充症状
        batch_qs = _attach_question_texts(batch_qs)
        batch_text = _build_question_text(batch_qs)

        # 节点内 multi-interrupt:body 重跑时 interrupt() 按调用顺序返已 cache 的 resume value
        user_answer = interrupt({
            "followup_questions": batch_qs,
            "followup_question": batch_text,
        })

        asked.update(batch_slots)
        accumulated_questions.extend(batch_qs)
        accumulated_answers.append(user_answer if isinstance(user_answer, str) else "")
        accumulated_texts.append(batch_text)

    # 全部 batch 收完 → 一次 LLM holistic 翻译(复用 ⑦ 的 parse 逻辑)
    # 推 custom event,把灰字从"正在准备需要确认的细节…"切到"正在处理回答…"
    # (节点 chain_start 只在 batch 收集阶段 fire 一次,要靠 custom event 加段)
    dispatch_custom_event("progress_step", {"step": "intake_translate_answers"})
    all_question = "\n\n".join(accumulated_texts)
    all_answer = "\n\n".join(accumulated_answers)
    update = parse_followup_response(
        followup_question=all_question,
        followup_answer=all_answer,
        questions=accumulated_questions,
        state=state,
    )
    return update
