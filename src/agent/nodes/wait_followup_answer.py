"""src/agent/nodes/wait_followup_answer.py — Agent ⑥ wait_followup_answer(DEV_SPEC §4.1.5)。

调 `langgraph.types.interrupt(...)` 暂停图执行,等待用户回答;恢复时只重执行本节点
(轻量),避免重复调 LLM(上游已生成的 followup_question 不会再调一次)。

**契约**:上游(④ / ⑤)进入本节点前已经把 `followup_question` 和 `followup_questions`
都填好(⑤ 是 LLM targeted 出题时同时拼;④ 是 select_symptom 模板拼),本节点零拼装。
"""
from __future__ import annotations

from langgraph.types import interrupt

from src.agent.state import MedicalState


def wait_followup_answer(state: MedicalState) -> dict:
    """interrupt 暂停;恢复时把用户回答写入 followup_answer。"""
    user_answer = interrupt(state.followup_question)
    return {"followup_answer": user_answer}
