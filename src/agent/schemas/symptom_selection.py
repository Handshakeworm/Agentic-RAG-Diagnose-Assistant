"""Agent ④ select_discriminative_symptom LLM 输出 schema(DEV_SPEC §9.5)。

④ 节点 1 次 LLM 调用直接选三件事:
- questions:追问项(slot 维度补全 / open 开放式),≤ MAX_FOLLOWUP_QUESTIONS
- unaskable_symptoms:LLM 自己想知道但患者答不上的体征/指标(粗筛),
  后续 ⑩ Step 3 会基于诊断结果再次校准,挑出"诊断后仍需检查确认的"写回
  state.unaskable_symptoms,⑧a recommend_exam 直接消费这份精筛版

`UnaskableSymptom` 同时被 ⑩ `DiagnosisOutput.retained_unaskable` 复用,字段
结构跨节点统一(spec §9.5 schema 6 / 9)。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FollowupQuestion(BaseModel):
    """④ select_symptom 单条追问项。"""

    type: Literal["slot", "open"] = Field(
        ...,
        description="slot=补全 13 维 HPI 空槽;open=开放式问'还有别的不舒服吗'",
    )
    slot: str | None = Field(
        None,
        description="type=slot 时填,如 'trigger' / 'location' / 'nature' 等 13 维槽位名;type=open 时为 None",
    )


class UnaskableSymptom(BaseModel):
    """LLM 想知道但患者答不上的体征/指标(④ 粗筛 + ⑩ 精筛共用)。

    ④ 出粗筛喂给 ⑩ Step 2 判 need_exam;⑩ Step 3 基于诊断结果挑出"仍需
    检查确认的"写回 state,⑧a 直接消费 description 作为检查建议来源。
    """

    description: str = Field(
        ...,
        description="医生侧语言:想查什么 / 想知道什么体征,如'腹部 B 超提示有无胆囊壁增厚'",
    )
    reason: str = Field(
        ...,
        description="为什么对鉴别诊断重要,如'关键鉴别胆囊炎 vs 胃炎'",
    )


class SmartFollowupOutput(BaseModel):
    """④ select_symptom LLM 输出。"""

    questions: list[FollowupQuestion] = Field(
        default_factory=list,
        max_length=5,
        description="本轮追问项列表(0-5 个);为空 = 信息已足,直接进诊断",
    )
    unaskable_symptoms: list[UnaskableSymptom] = Field(
        default_factory=list,
        max_length=5,
        description="LLM 想知道但患者答不上的体征/指标(0-5 条粗筛);为空 = 没有需检查鉴别的项",
    )
