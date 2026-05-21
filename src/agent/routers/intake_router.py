"""src/agent/routers/intake_router.py — Agent 入站/追问条件路由(DEV_SPEC §4.1.3)。

**纯函数路由**:仅返回下一节点名,不修改 State。

两个路由器:

1. `generate_followup_out_router`(在 ⑤ generate_followup 之后,3 路返回):
   优先级:askable(主观可问)> unaskable(客观需查体/检查)> 都空(进检索)
   - `followup_questions` 非空 → "to_wait" → ⑥ wait_followup_answer
   - 否则 `unaskable_symptoms` 非空 → "to_recommend_exam" → ⑧a 首诊模式
   - 都空 → "to_build_query" → ② build_query

2. `post_followup_router`(在 ⑦ process_followup_answer 之后):
   - **用 `candidate_chunks` 是否非空当"是不是已经检索过"的隐含信号**(不再依赖
     followup_source 元数据字段)
   - chunks 空(intake 后这条路, ⑤ → ⑥ → ⑦, 此时还没进 ②③)→ 回 ⑤ 让 LLM 再判
   - chunks 非空(④ 鉴别诊断追问完, ⑥ → ⑦, 已经过 ②③)→ 回 ② 重检索 → ④ 再判
   - 兜底:followup_round 触顶 → ② (防 LLM 死循环)
"""
from __future__ import annotations

from config.settings import settings
from src.agent.state import MedicalState


def generate_followup_out_router(state: MedicalState) -> str:
    """⑤ 出口(3 路):askable > unaskable > 空。

    askable 优先 — 主观题快、免费且为 ⑩ 提供文字证据;
    无 askable 但有 unaskable → ⑧a 首诊推单(让患者去医院做一次性查齐,⑩ 大概率一次结案);
    都空 → ② 直接进检索。
    """
    if state.followup_questions:
        return "to_wait"
    if state.unaskable_symptoms:
        return "to_recommend_exam"
    return "to_build_query"


def post_followup_router(state: MedicalState) -> str:
    """⑦ 出口:chunks 空(intake 后路径)回 ⑤ 再判;chunks 非空(④ 路径)回 ② 重检索。"""
    if state.followup_round >= settings.agent_limits.MAX_FOLLOWUP_ROUNDS:
        return "to_build_query"
    if not state.candidate_chunks:
        return "loop_to_followup"
    return "to_build_query"
