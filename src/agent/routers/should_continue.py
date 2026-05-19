"""src/agent/routers/should_continue.py — Agent 条件路由 should_continue(DEV_SPEC §4.1.3.1)。

**纯函数路由**:仅返回下一节点名,不修改 State。

优先级(spec §4.1.3.1):
1. followup_round >= MAX_FOLLOWUP_ROUNDS → "diagnose"(硬性兜底,防收敛失效;
   兜底 insufficient 产出由 Node ⑩ Step -1 完成,不在路由层做)
2. followup_questions 非空 → "followup"
3. 其他 → "diagnose"

④ 节点已内聚三重过滤(可问性/增益阈值/候选池耗尽),路由器只看 followup_questions 非空。
"""
from __future__ import annotations

from config.settings import settings
from src.agent.state import MedicalState


def should_continue(state: MedicalState) -> str:
    """纯函数路由,不写 State。"""
    if state.followup_round >= settings.agent_limits.MAX_FOLLOWUP_ROUNDS:
        return "diagnose"
    if state.followup_questions:
        return "followup"
    return "diagnose"
