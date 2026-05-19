"""Agent ⑦ process_followup_answer LLM 输出 schema(DEV_SPEC §9.5 第 7 项)。

⑦ 解析患者对 ⑤ 追问的回答,产出两类信息:
  - slot_fills: 维度级回填(对应 ④ 选出的 type=slot 追问项)
  - new_symptoms: 患者回答中主动提到的新症状(对应 ④ 选出的 type=open 追问项,
    或患者顺带补充的副症状),直接进 confirmed_symptoms 供下轮 build_query 使用
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FollowupParseResult(BaseModel):
    """⑦ process_followup_answer LLM 输出。"""

    slot_fills: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description=(
            "维度级回填,key=槽位名;value 类型与 PresentIllnessSlots 槽位一致:"
            "单值槽(onset_time/onset_mode/trigger/location/nature/severity/"
            "duration_pattern/progression/treatment_tried/treatment_response)为 str,"
            "多值槽(aggravating/relieving/associated_symptoms)为 list[str]"
        ),
    )
    new_symptoms: list[str] = Field(
        default_factory=list,
        description="患者回答中提及的新症状(开放式追问的主要产物,也包括患者顺带补充)",
    )
