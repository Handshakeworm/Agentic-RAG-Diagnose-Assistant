"""src/agent/nodes/wait_followup_answer.py — Agent ⑥ wait_followup_answer(DEV_SPEC §4.1.5)。

调 `langgraph.types.interrupt(...)` 暂停图执行,等待用户回答;恢复时只重执行本节点
(轻量),避免重复调 LLM(⑤ 已生成的 followup_question 不会再调一次)。
"""
from __future__ import annotations

from langgraph.types import interrupt

from src.agent.state import MedicalState


def wait_followup_answer(state: MedicalState) -> dict:
    """interrupt 暂停;恢复时把用户回答写入 followup_answer。"""
    user_answer = interrupt(state.followup_question)
    return {"followup_answer": user_answer}
