"""Agent ⑩ diagnose 1 步 LLM 输出 schema(DEV_SPEC §9.5 第 8 项)。

⑩ 重设计:3 步链 → 1 步(对齐评测口径 `.eval/rag_eval/run_diagnose_eval.py`)。
评测脚本一步 LLM + 信息全给已经能拿到 top1 93.5% / top3 100%,3 步链的证据归集 +
排序 + 校准是过度工程化,延迟显著(评测平均 2 分钟,3 步链会到 4-6 分钟)。改 1 步:
LLM 同时输出诊断排序 + retained_unaskable 精筛。

`RankedDisease.failure_reason` 由**节点代码**在兜底路径填充(spec §4.1.2 ⑩),
不由 LLM 输出。LLM 正常路径 failure_reason=None,异常兜底由 except 块构造。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.agent.schemas.symptom_selection import UnaskableSymptom


class RankedDisease(BaseModel):
    """单个候选疾病的诊断结果(字段对齐评测 `CandidateDiagnosis` + 生产新增 differentiation_type)。"""

    disease: str = Field(
        ...,
        description="疾病名;尽量精确到部位/分型(如 '右额颞急性硬膜外血肿' 而非 '颅内血肿');兜底场景固定为 '信息不足以支持可靠诊断'",
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="概率;兜底场景为 0.0"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="3-5 条关键支持证据(可引用症状/报告/文献/图像)",
    )
    differentiation: str | None = Field(
        None,
        description="与其他相似疾病的鉴别要点(可空)",
    )
    differentiation_type: Literal["confirmed", "need_exam", "insufficient"] = Field(
        ...,
        description="鉴别状态;top1 决定 router 走 ⑧ recommend_exam(need_exam)还是 ⑪ safety_gate(其他)",
    )
    failure_reason: str | None = Field(
        None,
        description=(
            "系统级失败原因(非自然 insufficient)。取值示例:"
            "'followup_round_capped'(追问触顶)、"
            "'step_1_structured_output_failed: ValidationError: ...'(LLM 结构化输出失败)。"
            "None 表示 LLM 正常推理。该字段由节点代码在兜底路径中填充,不由 LLM 输出。"
        ),
    )


class DiagnosisOutput(BaseModel):
    """⑩ diagnose 1 步 LLM 输出 — 诊断结果 + 精筛 unaskable。

    `retained_unaskable` 是 LLM 基于当前诊断结果产的"仍需检查确认"的 unaskable
    列表:**主要从上游 unaskable 粗筛(④ + ⑤ 累积)里挑/改写**,**也允许新产**
    (诊断推理后发现"上游没列但鉴别真需要"的检查项,如 MRCP / 特定标志物等)。
    节点代码写回 `state.unaskable_symptoms` 供 ⑧a recommend_exam 消费。
    LLM 判断不再需要的 → 不写进 retained_unaskable,自然丢弃。
    """

    results: list[RankedDisease] = Field(
        ..., min_length=1, description="按 probability 降序排列的诊断结果列表"
    )
    retained_unaskable: list[UnaskableSymptom] = Field(
        default_factory=list,
        description=(
            "基于诊断结果产的 unaskable 列表:主要从上游粗筛(④ + ⑤)挑/改写,"
            "也允许 LLM 基于诊断推理新产(但不要为加而加, 大多数 case 挑/改写够)。"
            "confirmed/insufficient 路径下可为空(不会被消费);need_exam 路径下"
            "应至少保留 1 条供 ⑧a 推荐检查"
        ),
    )
