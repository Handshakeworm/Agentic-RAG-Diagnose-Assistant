"""Agent intake_followup_ask LLM 输出 schema(DEV_SPEC §4.1.3 入站追问节点)。

intake_followup_ask 两阶段:
  - slot 阶段:零 LLM,按 13 维 HPI 空槽 + 始终一项 open 兜底
  - LLM 针对性阶段:13 维全填后**1 次 LLM 调用**,定还需追问哪些临床细节

`TargetedFollowupOutput` 仅约束 LLM 针对性阶段的输出:1~5 条追问问题,
**严格不输出诊断/疾病名/可能性/probability**(prompt 内显式禁止),
LLM 只负责追问,不抢 ⑩ diagnose 的活。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TargetedFollowupItem(BaseModel):
    """1 条针对性追问。"""

    question: str = Field(
        ...,
        description="自然语言完整问句,如'有没有发热?最高多少度?'(可串联同主题的小问)",
    )
    target: str = Field(
        ...,
        description="本问询所靶向的临床信息名,如'fever' / 'radiation' / 'duration_aggravation';仅用于审计/调试,不需翻译成中文",
    )


class TargetedFollowupOutput(BaseModel):
    """intake_followup_ask LLM 针对性阶段输出。"""

    questions: list[TargetedFollowupItem] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "0~5 条仍需向患者补充询问的临床信息。"
            "为空 = 已充分,直接转 done 进 ②。"
            "**禁止输出诊断/疾病名/可能性/probability;LLM 只问不诊。**"
        ),
    )
