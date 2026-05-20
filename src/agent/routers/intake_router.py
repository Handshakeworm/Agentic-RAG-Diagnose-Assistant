"""src/agent/routers/intake_router.py — Agent 入站/追问条件路由(DEV_SPEC §4.1.3)。

**纯函数路由**:仅返回下一节点名,不修改 State。

两个路由器:

1. `generate_followup_out_router`(在 ⑤ generate_followup 之后):
   - 看 state.followup_questions 是否非空
   - 非空 → 走 ⑥ wait_followup_answer(让用户答 form)
   - 空 → 走 ② build_query(intake 阶段 LLM 判够了 / 无追问任务)

2. `post_followup_router`(在 ⑦ process_followup_answer 之后):
   - 看 state.followup_source(intake 节点末尾设 "intake",④ 末尾设 "diagnostic")
   - "intake" → 回 ⑤(让 ⑤ LLM 再判要不要 targeted)
   - "diagnostic" → 走 ② build_query(④ 鉴别诊断追问答完 → 回检索 → ④ 再判)
   - 兜底:followup_round 触顶 → ② (防 LLM 死循环)
"""
from __future__ import annotations

from config.settings import settings
from src.agent.state import MedicalState


def generate_followup_out_router(state: MedicalState) -> str:
    """⑤ 出口:有题 → ⑥ 让用户答;无题 → ② 直接进检索。"""
    if state.followup_questions:
        return "to_wait"
    return "to_build_query"


def post_followup_router(state: MedicalState) -> str:
    """⑦ 出口:intake 阶段回 ⑤ 让 LLM 再判;diagnostic 阶段回 ② 重检索。"""
    if state.followup_round >= settings.agent_limits.MAX_FOLLOWUP_ROUNDS:
        return "to_build_query"
    if state.followup_source == "intake":
        return "loop_to_followup"
    return "to_build_query"
