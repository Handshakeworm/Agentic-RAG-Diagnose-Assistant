"""src/agent/state.py — LangGraph StateGraph 共享状态(DEV_SPEC §4.1.1 / §4.1.1a)。

`MedicalState` 是 Pydantic BaseModel,37 字段分 9 段(消息 / 患者 / 术语 / 召回 /
追问 / 诊断 / 安全 / 建议 / 审计)。

**为什么 Pydantic 而非 TypedDict**:§9.2 兼容性规则全是 Pydantic Field 语法
(checkpointer 反序列化老 state 时缺字段自动填默认值,避免 KeyError);§9.5 inner
schema 已用 Pydantic;嵌套结构(token_usage / latency / present_illness_slots)
强类型化把"注释式 schema"升级为 runtime 强制约束。

**字段权威定义** → DEV_SPEC §4.1.1 / §4.1.1a
**演进规则**     → DEV_SPEC §9.2(加字段必须给默认,不收窄类型)
**审计字段对照** → DEV_SPEC §9.6.2(`session_token_usage` / `session_latency_ms` /
                  `last_reranked_chunks` / `last_diagnose_prompt` / `last_diagnose_raw_output`)

**调用方式**(典型):

    from src.agent.state import MedicalState, create_initial_state
    initial = create_initial_state(patient_id="P001", patient_input="头疼一周")
    result = graph.invoke(initial)

注:`create_initial_state` 现在是薄 wrapper,Pydantic 默认值已接管大部分构造逻辑;
保留工厂是为了显式传必填字段 + 与 spec §4.1.1a 调用约定保持一致。
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


# ────────────────────────────────────────────────────────────────────────────
# 哨兵常量:slot 被问过但患者明确答"不知道/没注意/没有"
# ────────────────────────────────────────────────────────────────────────────
# 由 ⑦ process_followup_answer 的 LLM 在 slot_fills 中写入(单值=字符串,多值=单元素 list)。
# 语义:已问过 ≠ 未询问;不进检索 query(BM25/dense 噪音),但在诊断 prompt 里保留(医学上
# "患者明确说没诱因"是有效信号)。所有 sparse_queries / dense_query 构建处必须过滤掉它。
SLOT_UNKNOWN_SENTINEL = "(患者未明确)"


# ────────────────────────────────────────────────────────────────────────────
# 嵌套子模型:把 spec 注释里的"内联 schema"提升为强类型 BaseModel
# ────────────────────────────────────────────────────────────────────────────


class PresentIllnessSlots(BaseModel):
    """§4.1.1 现病史 13 维结构化槽位(3 个多值 list[str] + 10 个 str | None)。"""

    onset_time: str | None = None
    onset_mode: str | None = None
    trigger: str | None = None
    location: str | None = None
    nature: str | None = None
    severity: str | None = None
    duration_pattern: str | None = None
    aggravating: list[str] = Field(default_factory=list)
    relieving: list[str] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    progression: str | None = None
    treatment_tried: str | None = None
    treatment_response: str | None = None


class SessionTokenUsage(BaseModel):
    """§9.6.2 `token_usage` 字段;由 §9.1 RetryObserver.on_llm_end 累加。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class SessionLatencyMs(BaseModel):
    """§9.6.2 `latency_ms` 字段;`total` 由 API 层求和,不在此处维护。"""

    intent: int = 0
    retrieval: int = 0
    rerank: int = 0
    llm_call: int = 0
    post_process: int = 0


# ────────────────────────────────────────────────────────────────────────────
# MedicalState 主模型(LangGraph StateGraph schema)
# ────────────────────────────────────────────────────────────────────────────


class MedicalState(BaseModel):
    """LangGraph 共享状态。字段语义见 DEV_SPEC §4.1.1。

    注:`diagnosis_result` / `report_findings` 等 inner item 仍用 `list[dict]` 表示,
    待 §9.5 对应 Pydantic schema 实现后(F4-F10 阶段)升级为 `list[RankedDisease]` /
    `list[ReportFinding]` 等强类型。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # BaseMessage 不是 Pydantic 类型

    # === 消息历史 ===
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # === 患者信息 ===
    patient_id: str
    patient_input: str
    chief_complaint: str = ""
    present_illness: str = ""
    present_illness_slots: PresentIllnessSlots = Field(default_factory=PresentIllnessSlots)
    medical_history: dict = Field(default_factory=dict)
    exam_reports: list[dict] = Field(default_factory=list)
    report_findings: list[dict] = Field(default_factory=list)

    # === 召回与候选 ===
    dense_query: str = ""
    sparse_queries: list[str] = Field(default_factory=list)
    candidate_chunks: list[dict] = Field(default_factory=list)
    confirmed_symptoms: list[str] = Field(default_factory=list)
    denied_symptoms: list[str] = Field(default_factory=list)
    uncertain_symptoms: list[str] = Field(default_factory=list)

    # === 追问控制 ===
    followup_round: int = 0
    last_nlu_round: int = 0
    followup_question: str = ""
    followup_answer: str = ""
    followup_questions: list[dict] = Field(default_factory=list)
    unaskable_symptoms: list[dict] = Field(default_factory=list)
    exam_round: int = 0
    pending_exam_results: list = Field(default_factory=list)

    # === 入站追问状态机(intake_followup 节点专用) ===
    # "slot" 阶段:循环按 13 维 HPI 空槽 + open 兜底追问;
    # "done" 阶段:slot 已填满且 LLM 针对性追问已发出(或失败兜底),允许 router 放行到 ②。
    intake_phase: Literal["slot", "done"] = "slot"

    # === ⑦ 后路由分流:决定 ⑤/⑥/⑦ 走完是回 ⑤(intake targeted loop)还是 ②(④ 鉴别诊断回检索) ===
    # intake 节点末尾写 "intake";④ select_symptom 写好 followup_questions 时写 "diagnostic"
    followup_source: Literal["intake", "diagnostic"] | None = None

    # === 诊断结果 ===
    diagnosis_result: list[dict] = Field(default_factory=list)

    # === 安全约束(safety_gate ⑪ 产出)===
    safety_constraints: dict = Field(default_factory=dict)

    # === 建议输出 ===
    recommended_tests: list[str] = Field(default_factory=list)
    medication_advice: list[dict] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    final_response: str = ""

    # === 审计埋点(rag_trace 写入,见 §9.6.2)===
    last_reranked_chunks: list[dict] = Field(default_factory=list)
    session_token_usage: SessionTokenUsage = Field(default_factory=SessionTokenUsage)
    session_latency_ms: SessionLatencyMs = Field(default_factory=SessionLatencyMs)
    last_diagnose_prompt: str | None = None
    last_diagnose_raw_output: str | None = None


def create_initial_state(patient_id: str, patient_input: str) -> MedicalState:
    """显式传必填字段构造初始 State(§4.1.1a 调用约定)。

    Pydantic 默认值机制接管了大部分字段构造,工厂只负责确保两个必填项被显式传入。
    """
    return MedicalState(patient_id=patient_id, patient_input=patient_input)
