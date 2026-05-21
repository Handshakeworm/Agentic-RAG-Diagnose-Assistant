"""Agent ⑤ generate_followup LLM 输出 schema(DEV_SPEC §4.1.2 ⑤ / §9.5)。

⑤ 是检索前 holistic gate,**拆 2 步 LLM 调用**:

- **Step A(pro reasoner 决策)**:`HolisticGateDecision`
  - askable_targets: list[str]  (英文短标签,患者侧主观靶点)
  - unaskable_findings: list[UnaskableSymptom]  (查体/检查项)
  - 只出"决策",不出"完整问句";thinking 模型 output token 大幅减少

- **Step B(flash 拼问句,仅 askable_targets 非空时调)**:`QuestionGenOutput`
  - questions: list[TargetedFollowupItem]  (把短标签 → 患者侧自然中文问句)
  - 非 thinking 模型,1-3 秒完成

`UnaskableSymptom`(`{description, reason}`)从 `symptom_selection` 复用 — ④/⑤/⑩
三处都用同款。`TargetedFollowupItem`(`{question, target}`)是 ⑤ 最终给患者的
自然语言完整问句 + 英文短标签(target 仅审计/调试,前端不渲染)。

**严格不输出诊断/疾病名/probability/differential**(prompt 内显式禁止),
LLM 只判"还差什么 + 该问还是该查",不抢 ⑩ diagnose 的活。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.schemas.symptom_selection import UnaskableSymptom


class TargetedFollowupItem(BaseModel):
    """1 条针对性追问(患者主观能答)。"""

    question: str = Field(
        ...,
        description="自然语言完整问句,如'有没有发热?最高多少度?'(可串联同主题的小问)",
    )
    target: str = Field(
        ...,
        description="本问询所靶向的临床信息名,如'fever' / 'radiation' / 'duration_aggravation';仅用于审计/调试,不需翻译成中文",
    )


class HolisticGateDecision(BaseModel):
    """⑤ Step A(pro 决策)LLM 输出:精简 schema,只出"决策",不出"完整问句"。

    Step A 用 pro(reasoner)做 holistic 推理"还差什么 + 该问还是该查";
    askable 部分只输出英文短标签(节省 thinking 模型的 output token 耗时),
    完整自然问句留给 Step B(flash)拼。

    优先级(节点内 generate_followup 处理):
      askable_targets 非空 → 调 Step B 拼问句 → ⑥
      否则 unaskable_findings 非空 → ⑧a 首诊
      两 list 都空 → ②
    """

    askable_targets: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "0~5 条仍需向患者补充询问的**靶点短标签**(英文,患者侧主观信息)。"
            "如 fever_max_temp / radiation_to_back / aggravating_after_meal / "
            "associated_jaundice。**只写靶点名,不写完整问句**(Step B 由 flash 补)。"
            "为空 = 主观信息已充分。**禁止输出诊断/疾病名/可能性。**"
        ),
    )
    unaskable_findings: list[UnaskableSymptom] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "0~5 条需要通过**查体或检查**才能获知的客观证据(患者主观答不上的)。"
            "如 Murphy 征 / T3 数值 / 腹部 B 超 / 心电图等。"
            "为空 = 不需要新做查体/检查。"
            "**写医生侧语言**(description=要查什么 / reason=为什么对鉴别重要)。"
        ),
    )


class QuestionGenOutput(BaseModel):
    """⑤ Step B(flash 拼问句)LLM 输出。

    输入 askable_targets list,把每个短标签转成患者能听懂的自然中文问句;
    保留 target 字段做 1-1 映射(审计/调试用)。Step A 的 target 顺序应保留。
    """

    questions: list[TargetedFollowupItem] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "把输入的每个 askable_target 转成 1 条 {question, target} —— "
            "question 是患者侧自然中文问句(口语化,可串联同主题的小问),"
            "target 必须**原样**回填输入的短标签(便于审计 1-1 映射)。"
        ),
    )
