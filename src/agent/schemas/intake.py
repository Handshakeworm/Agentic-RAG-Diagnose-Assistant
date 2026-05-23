"""Agent ⑤ generate_followup LLM 输出 schema(DEV_SPEC §4.1.2 ⑤ / §9.5)。

⑤ 是检索前 holistic gate,**拆 2 步 LLM 调用**:

- **Step A(flash 决策)**:`HolisticGateDecision`
  - askable_targets: list[str]  (**中文短语**,患者侧主观靶点)
  - unaskable_findings: list[UnaskableSymptom]  (查体/检查项)
  - 只出"决策",不出"完整问句"

- **Step B(flash 拼问句,仅 askable_targets 非空时调)**:`QuestionGenOutput`
  - questions: list[TargetedFollowupItem]  (把中文短语 → 患者侧自然中文问句)
  - 非 thinking 模型,1-3 秒完成

`UnaskableSymptom`(`{description, reason}`)从 `symptom_selection` 复用 — ④/⑤/⑩
三处都用同款。`TargetedFollowupItem`(`{question, target}`)是 ⑤ 最终给患者的
自然语言完整问句 + 中文短语标签(target 是 askable_target 原样回填,用于审计 + 让
下一轮 Step A 直接字符串去重对账,中英不再来回翻译)。

**2026-05-22**:askable_target 从英文 snake_case 改中文短语 —— 因为 denied/confirmed
是中文,英文 target 让 Step A 心里翻译对账常漏,统一中文后直接字符串对照。
专业医学术语(Murphy 征 / WBC / NRS 评分 / ECG / B 超)允许英文/缩写。

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
        description=(
            "本问询所靶向的临床信息**中文短语**,如'发烧最高温度' / '疼痛放射部位' / "
            "'加重因素与饭后关系' / '胆石家族史' / '近期饮酒'。"
            "用于审计 + 让下一轮 Step A 直接字符串去重对账。"
            "专业医学缩写允许英文(Murphy 征 / NRS 评分)。"
        ),
    )


class HolisticGateDecision(BaseModel):
    """⑤ Step A(flash 决策)LLM 输出:精简 schema,只出"决策",不出"完整问句"。

    Step A 做 holistic 推理"还差什么 + 该问还是该查";askable 部分只输出
    中文短语(便于下一轮直接跟 denied/confirmed 中文列表对账去重),
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
            "0~5 条仍需向患者补充询问的**中文短语靶点**(患者侧主观信息)。"
            "如 '发烧最高温度' / '疼痛放射到背部' / '饭后是否加重' / '皮肤巩膜黄染' / "
            "'胆石家族史' / '近期饮酒史'。**只写靶点名,不写完整问句**(Step B 由 flash 补)。"
            "为空 = 主观信息已充分。**禁止输出诊断/疾病名/可能性。** "
            "专业医学缩写允许英文(NRS 评分 / WBC 等)。"
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

    输入 askable_targets list(中文短语),把每个转成患者能听懂的自然中文问句;
    保留 target 字段做 1-1 映射(target = 原中文短语,便于下一轮 Step A 去重对账)。
    """

    questions: list[TargetedFollowupItem] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "把输入的每个 askable_target(中文短语)转成 1 条 {question, target} —— "
            "question 是患者侧自然中文问句(口语化,可串联同主题的小问,但**不能串入** "
            "已 denied/confirmed 的症状),"
            "target 必须**原样**回填输入的中文短语(便于审计 1-1 映射 + 下轮去重)。"
        ),
    )
