# 9. 全局实现契约（跨章节）

本章定义**跨章节生效的工程契约**，区别于第 1～8 章聚焦各自业务职责（what）、第 9 章定义所有章节共同遵守的实现约束（how）。

**本章的特殊地位（实现前必读）**：

- 本章约束**不绑定单一业务章节**，横跨数据摄取（第 3 章）、Agent 工作流（第 4 章）、评估系统（第 6 章）、Prompt 模板（第 7 章）等多个实现领域
- 按章节拆分加载 spec 时，**任何涉及 LLM 调用、Schema 定义或 Pydantic 结构化输出的任务**，除了读本任务所在业务章节外，**必须同时读本章**
- 本章若与业务章节描述出现冲突，**以本章为准**（业务章节负责 what，本章负责 how）

---

## 9.1 统一机制

系统中有 20+ 个 LLM 调用点需要返回结构化数据（JSON / 固定字段），而非自由文本。本节统一定义保障机制，各业务章节的节点设计中不再重复说明。

**默认方案**：所有需要结构化输出的 LLM 调用均通过 LangChain 的 `llm.with_structured_output(PydanticModel)` 约束。底层走 DashScope OpenAI-compatible API 的 `response_format: {"type": "json_schema", "json_schema": ...}`，由模型原生 JSON Mode 保证输出合规，不依赖正则解析或后处理。

**Schema 定义位置**：Pydantic Schema 集中定义在 `src/agent/schemas/` 下，按职责分文件：

```
src/agent/schemas/
├── __init__.py
├── info_collect.py          # InfoCollectOutput
├── report_parser.py         # ReportFinding, ReportFindings
├── ner.py                   # NEREntity, NERResult
├── query_construction.py    # QueryConstructionOutput
├── symptom_selection.py     # FollowupQuestion, SmartFollowupOutput
├── followup.py              # FollowupParseResult
├── diagnosis.py             # RankedDisease, DiagnosisOutput（完整定义见 §9.5）
├── safety_gate.py           # SafetyGateOutput
├── advice.py                # AdviceOutput
├── ingestion.py             # ChunkEnrichmentOutput
└── evaluation.py            # 各 LLM Judge 评分 Schema
```

> 注：`src/api/schemas/` 存放 API 层的 HTTP 请求/响应模型（面向前端），与此处的 LLM 输出 Schema（面向模型）职责不同，不混用。

**校验失败分级处理**：

| 安全等级 | 最多尝试次数 | 失败后行为 | 适用场景 |
|---------|------------|-----------|---------|
| **高**（影响诊断安全） | 3（首次 + 2 次重试，对应 `stop_after_attempt=3`） | 产出兜底保守值，确保流水线不中断、结果偏保守而非偏激进 | ⑩ 诊断推理、⑪ 安全门控 |
| **中**（影响流水线流转） | 3（首次 + 2 次重试，对应 `stop_after_attempt=3`） | 抛 `StructuredOutputError`，由 StateGraph 错误处理捕获并终止当前会话，返回用户友好错误提示 | ①②⑤⑦⑧⑫ 等核心 Agent 节点 |
| **低**（不影响主流程） | 2（首次 + 1 次重试，对应 `stop_after_attempt=2`） | 跳过当前项 / 降级为无增强数据继续 | enrichment（跳过该 chunk 增强）、evaluation（标记该 case 评估失败） |

实现模式（伪代码）：

以下示例展示每个安全等级的完整实现模板（含裸代码指标上报）。`src/common/metrics.py` 模块级声明 6 个指标对象和 `retry_observer`，各调用点 import 使用。

```python
# src/common/metrics.py（模块级声明，所有调用点共享）
from prometheus_client import Counter, Histogram
from langchain_core.callbacks import BaseCallbackHandler

_attempts = Counter("structured_output_attempt_total", "...", ["node", "schema"])
_retries  = Counter("structured_output_retry_total",   "...", ["node", "schema"])
_failures = Counter("structured_output_failure_total", "...", ["node", "schema", "exception_type"])
_fallbacks = Counter("structured_output_fallback_triggered_total", "...", ["node", "fallback_type"])
_latency  = Histogram("structured_output_latency_seconds", "...", ["node", "schema"])
_diagnose_reason = Counter("diagnose_failure_reason_total", "...", ["reason_kind"])

class RetryObserver(BaseCallbackHandler):
    """捕获 with_retry 内部的重试事件（LangChain 框架原生扩展点，不是本项目封装层）。"""
    def on_retry(self, retry_state, *, metadata=None, **kwargs):
        md = metadata or {}
        _retries.labels(
            node=md.get("node", "unknown"),
            schema=md.get("schema", "unknown"),
        ).inc()

retry_observer = RetryObserver()
```

```python
# ── 中安全等级 — 各业务节点典型调用（以 ② build_query Step 1 NER 为例）──
from src.common.metrics import _attempts, _failures, _latency, retry_observer

node, schema_name = "build_query_step1_ner", "NERResult"
_attempts.labels(node=node, schema=schema_name).inc()
t0 = time.perf_counter()
try:
    chain = llm.with_structured_output(NERResult).with_retry(stop_after_attempt=3)
    ner_result = chain.invoke(
        ner_prompt,
        config={"callbacks": [retry_observer], "metadata": {"node": node, "schema": schema_name}},
    )
except Exception as e:
    _failures.labels(node=node, schema=schema_name, exception_type=type(e).__name__).inc()
    logger.error(f"[{node}] structured output failed: {e}", exc_info=True)
    raise  # "中"级：重试耗尽后向上抛异常，由 StateGraph 捕获终止会话
finally:
    _latency.labels(node=node, schema=schema_name).observe(time.perf_counter() - t0)

# ── 高安全等级（⑩ diagnose 1 步 LLM）——失败兜底 + failure_reason 记录 ──
# 1 步 LLM(对齐 RAG 评测口径,3 步链已废弃);失败即走 insufficient 兜底。
from src.common.metrics import _attempts, _failures, _latency, _fallbacks, _diagnose_reason, retry_observer

node, schema_name = "diagnose", "DiagnosisOutput"
chain = vision_llm.with_structured_output(DiagnosisOutput).with_retry(stop_after_attempt=3)

_attempts.labels(node=node, schema=schema_name).inc()
t0 = time.perf_counter()
try:
    result = chain.invoke(
        diagnose_messages,
        config={"callbacks": [retry_observer], "metadata": {"node": node, "schema": schema_name}},
    )
except Exception as e:
    _failures.labels(node=node, schema=schema_name, exception_type=type(e).__name__).inc()
    logger.error(f"diagnose failed: {type(e).__name__}: {e}", exc_info=True)
    # 业务层指标：fallback 触发 + failure_reason 分类
    _fallbacks.labels(node="diagnose", fallback_type="insufficient").inc()
    _diagnose_reason.labels(reason_kind="step_1_failed").inc()
    result = DiagnosisOutput(results=[RankedDisease(
        disease="信息不足以支持可靠诊断",
        probability=0.0,
        evidence=["Step 1 结构化输出失败"],
        differentiation=None,
        differentiation_type="insufficient",
        failure_reason=f"step_1_structured_output_failed: {type(e).__name__}: {e}",
    )])
finally:
    _latency.labels(node=node, schema=schema_name).observe(time.perf_counter() - t0)

# ── 高安全等级（⑪ safety_gate LLM 兜底，单步无下游依赖）——保守提示 ──
node, schema_name = "safety_gate_llm", "SafetyGateOutput"
_attempts.labels(node=node, schema=schema_name).inc()
t0 = time.perf_counter()
try:
    chain = llm.with_structured_output(SafetyGateOutput).with_retry(stop_after_attempt=3)
    result = chain.invoke(
        prompt,
        config={"callbacks": [retry_observer], "metadata": {"node": node, "schema": schema_name}},
    )
except Exception as e:
    _failures.labels(node=node, schema=schema_name, exception_type=type(e).__name__).inc()
    _fallbacks.labels(node=node, fallback_type="safety_conservative").inc()
    # 保守路径：视为"无法排除风险"，在 safety_constraints 追加通用警告
    result = SafetyGateOutput(additional_risks=[FALLBACK_SAFETY_UNAVAILABLE])
finally:
    _latency.labels(node=node, schema=schema_name).observe(time.perf_counter() - t0)

# ── 低安全等级 — 跳过降级（以 3.1.3 enrichment 为例）──
node, schema_name = "enrichment", "ChunkEnrichmentOutput"
_attempts.labels(node=node, schema=schema_name).inc()
t0 = time.perf_counter()
try:
    chain = llm.with_structured_output(ChunkEnrichmentOutput).with_retry(stop_after_attempt=2)
    result = chain.invoke(
        prompt,
        config={"callbacks": [retry_observer], "metadata": {"node": node, "schema": schema_name}},
    )
except Exception as e:
    _failures.labels(node=node, schema=schema_name, exception_type=type(e).__name__).inc()
    _fallbacks.labels(node=node, fallback_type="skip").inc()
    logger.warning(f"Chunk {chunk_id} enrichment failed, skipping: {e}")
    result = None  # 该 chunk 无增强元数据，不影响其他 chunk
finally:
    _latency.labels(node=node, schema=schema_name).observe(time.perf_counter() - t0)
```

> 以上模板每个 LLM 调用点约 10-15 行样板，20+ 调用点合计约 300 行重复代码。这是"不做封装"的明确代价，换取**零抽象风险**（异常作用域清晰、重试可观察、无签名约束）。实现各调用点时按 §9.3 清单选对 node 名和 schema 名，复制模板即可。

**实现风格约定（重要）**：

本项目**不使用装饰器或 helper 函数**封装结构化输出调用。每个 LLM 调用点独立裸写，理由：
- 装饰器/helper 会引入签名约束、异常作用域模糊、与 `with_retry` 内部行为不可见等问题
- 20+ 调用点的"重复样板"总量约 300 行，作为一致性代价可接受
- 一致性由**文档 + Schema 契约（§9.3）+ code review** 保障，而非代码抽象

**`src/common/metrics.py` 的职责仅限**：
1. 模块级声明 6 个 Prometheus 指标对象（单例，禁止在调用处重新创建）
2. 声明一个 LangChain `BaseCallbackHandler` 子类 `RetryObserver` 用于捕获 `with_retry` 内部重试事件
3. 除此之外**不提供任何 LLM 调用封装**

**可观测性要求（指标上报主体明确分工）**：

每个 LLM 调用点必须在业务代码内按下表主动上报以下 6 个指标（定义见 4.2.7 "结构化输出健康度"表）：

| 指标 | 上报主体 | 上报时机 |
|------|---------|--------|
| `structured_output_attempt_total` | 业务代码 | 进入 try 块前 `.labels(node, schema).inc()` |
| `structured_output_retry_total` | `RetryObserver` callback | 业务代码 invoke 时传入 `config={"callbacks": [retry_observer], "metadata": {"node": ..., "schema": ...}}`；callback 在 `on_retry` 内 `.inc()` |
| `structured_output_failure_total` | 业务代码 | `except` 分支内 `.labels(node, schema, exception_type=type(e).__name__).inc()` |
| `structured_output_fallback_triggered_total` | 业务代码 | 执行兜底路径前 `.labels(node, fallback_type).inc()` |
| `structured_output_latency_seconds` | 业务代码 | `try/except/finally` 内用 `time.perf_counter()` 差值 `.observe()` |
| `diagnose_failure_reason_total` | 业务代码（⑩ diagnose 专属） | 写入 `diagnosis_result[0].failure_reason` 时按 `reason_kind` 分桶 `.inc()`；取值：`followup_round_capped` / `step_1_failed`（⑩ 重设计为 1 步 LLM 后,3 步链废弃,step_2/3_failed 不再产出） |

> **使用 LangChain Callback 不等于"引入抽象"**：`RetryObserver` 继承 `BaseCallbackHandler`，是 LangChain 框架原生扩展点（类比 logger），不是本项目自建的封装层。`with_retry` 内部重试发生在 LangChain Runnable 内部，调用边界看不到——这是用 callback 而非 try/except 捕获的唯一原因。

---

## 9.2 Schema 演进兼容性

Schema 字段一旦上线即进入两个长生命周期消费路径，**不允许做破坏性变更**：
1. **Checkpointer 持久化的 State**:中断会话恢复时,旧 State 里的 `list[dict]`(如 `diagnosis_result`、`report_findings`)会用当前 Schema 反序列化。旧数据缺新字段 → Pydantic 抛 `ValidationError` → 会话无法恢复。
2. **审计表 `rag_trace.retrieved_chunks` / `diagnosis_feedback.expected_response` 等 JSONB 字段**：历史记录用旧 Schema 写入，读取做分析 / 回归测试时走当前 Schema 解析。

兼容性规则（新增字段时必须遵守）：

| 场景 | 必须 | 禁止 |
|------|------|------|
| 加新字段 | `Field(None, ...)` 或 `Field(default_factory=list/dict, ...)` 提供默认值 | `Field(..., description=...)` 将新字段设为必填 |
| 改字段类型 | 兼容类型放宽（`str` → `str \| None`、`int` → `int \| float`） | 收窄类型（`str \| None` → `str`）、改语义（`confidence: float` 从概率改为对数似然） |
| `Literal[...]` 枚举 | 只允许**新增**取值 | 删除已有取值、重命名取值 |
| 删字段 | 保留字段并标记 `deprecated=True`（Pydantic v2 支持），运行时忽略 | 直接物理删除 |

破坏性变更的正确做法：**新开一个 Schema 版本**（如 `DiagnosisOutputV2`），Node ⑩ 按版本号分发，保持 V1 至少存活一个迁移周期直到所有旧 checkpointer 自然过期（参考 5.2.3.5 `rag_trace` 保留 90 天）。

---

## 9.3 全量结构化输出清单

> 各 Schema 的完整 Pydantic 类定义（含子模型、字段约束）见 §9.5，下表仅列关键字段供快速查阅。
>
> **清单维护规则**：任何新增 LLM 调用点必须先补充到本清单（含 Schema 名 / 关键字段 / 安全等级 / 失败处理），再动手实现业务代码。清单中不存在的调用点视为违规实现。

**一、Agent 核心流水线（对应第 4 章）**

| 调用点 | Schema | 关键字段 | 安全等级 | 失败处理 |
|-------|--------|---------|---------|---------|
| ① `info_collect` Step 1 | `InfoCollectOutput` | `chief_complaint: str`, `present_illness: str`, `present_illness_slots: dict`（13 个维度槽位，未提及维度为 None/空列表） | 中 | 最多尝试 3 次；仍失败则抛异常终止会话（无主诉无法继续） |
| ①.5 `analyze_initial_reports` / ⑨ `process_exam_result` | `ReportFindings` | `findings: list[ReportFinding]`；每项含 `report_type: str`, `abnormal_values: list[str]`, `impressions: list[str]`, `positive_findings: list[str]`, `negative_findings: list[str]` | 中 | 最多尝试 3 次；仍失败则该份报告标记解析失败，`report_findings` 不追加该项，流水线继续（降级为无该报告证据） |
| ② `build_query` Step 1 NER | `NERResult` | `entities: list[NEREntity]`;每项含 `text: str`, `entity_type: Literal["symptom","disease","drug","anatomy"]`, `negation: bool`, `temporality: Literal["current","past","family"]`, `value: str｜None` | 中 | 最多尝试 3 次;仍失败则抛异常 |
| ② `build_query` Step 3 Query 构建 | `QueryConstructionOutput` | `dense_query: str`(单字段;sparse_queries 由 Step 2 确定性产出,不进 LLM 输出) | 中 | 最多尝试 3 次;仍失败则抛异常 |
| ⑤ `select_symptom` 智能追问选择 | `SmartFollowupOutput` | `questions: list[FollowupQuestion]`(≤ MAX_FOLLOWUP_QUESTIONS);每项 `type: Literal["slot","open"]` + `slot: str\|None`;`unaskable_symptoms: list[UnaskableSymptom]`(≤ MAX_FOLLOWUP_QUESTIONS);每项 `description: str` + `reason: str`(粗筛版,⑩ Step 3 会精筛覆盖) | 中 | 最多尝试 3 次;仍失败则返回空 questions + 空 unaskable → `should_continue` 路由跳诊断 |
| ⑦ `process_followup_answer` | `FollowupParseResult` | `symptom_responses: list[dict]`（每项含 `term: str`, `status: Literal["confirmed","denied","uncertain","unanswered"]`）, `slot_fills: dict[str, str \| list[str]]`（维度级回填，单值槽 str / 多值槽 list[str]，与 `PresentIllnessSlots` 类型对齐）, `new_symptoms: list[str]` | 中 | 最多尝试 3 次；仍失败则抛异常（追问回答未解析将导致信息丢失） |
| ⑩ `diagnose` 1 步 LLM（**原生多模态模型** — `settings.llm.VISION_BASE_URL` / `VISION_API_KEY` / `VISION_MODEL_NAME`，DashScope qwen3.5-plus） | `DiagnosisOutput` | `results: list[RankedDisease]`（每项 disease / probability / evidence / differentiation / differentiation_type / failure_reason）+ `retained_unaskable: list[UnaskableSymptom]`（精筛覆盖 ⑤ 粗筛 → 写回 `state.unaskable_symptoms` 供 ⑧a 消费）；context 含 figure 时 `image_path` 转 base64 作为多模态消息送入（详见 §3.2.3 LLM 路由段）；完整定义见 §9.5 | 高 | 最多尝试 3 次；失败兜底产出 insufficient 结果并在 `failure_reason` 字段记录 `"step_1_structured_output_failed: <ExcType>: <msg>"`（详见 4.1.2 ⑩ 结构化输出保障） |
| ⑧a `recommend_exam` | `RecommendExamOutput` | `tests: list[str]`（每项一个检查名，如"血常规"/"腹部 CT"，期望 3-5 项）, `rationale: str`（整体说明，2-3 句） | 中 | 最多尝试 3 次；仍失败则抛异常终止会话（检查推荐失败说明 LLM 完全不可用） |
| ⑪ `safety_gate` LLM 兜底 | `SafetyGateOutput` | `additional_risks: list[dict]`（每项含 `risk_type: Literal["cross_allergy","interaction","dosage_adjustment"]`, `description: str`, `severity: Literal["high","medium","low"]`, `recommendation: str`） | 高 | 最多尝试 3 次；仍失败则走保守路径——LLM 兜底层视为"无法排除风险"，在 `safety_constraints` 中追加通用警告："LLM 安全评估不可用，建议线下由药师复核" |
| ⑫ `generate_advice` | `AdviceOutput` | `medications: list[dict]`, `exam_suggestions: list[str]`, `risk_warnings: list[str]`, `urgent_flag: bool` | 中 | 最多尝试 3 次；仍失败则抛异常 |

**二、数据摄取层（对应第 3 章）**

| 调用点 | Schema | 关键字段 | 安全等级 | 失败处理 |
|-------|--------|---------|---------|---------|
| 3.1.3 `enrichment` | `ChunkEnrichmentOutput` | `title: str`, `summary: str`, `hypothetical_questions: list[str]` | 低 | 最多尝试 2 次；仍失败则跳过该 chunk 的增强，`title`/`summary`/`hypothetical_questions` 留空，chunk 仅以 `original_content` 参与检索（精度降低但不丢数据） |

> 说明：3.2.1 查询处理（关键词识别 / 术语扩展 / Dense Query 整合改写）所有 LLM 调用均在 Agent ② `build_query` 节点内完成，对应 Schema 见上方"一、Agent 核心流水线"中 ② 的 4 个 Step；不再单列"查询处理层"。

**三、离线评估层（对应第 6 章）**

| 调用点 | Schema | 关键字段 | 安全等级 | 失败处理 |
|-------|--------|---------|---------|---------|
| `build_rag_faithfulness_prompt` | `FaithfulnessScore` | `claims: list[dict]`（每条陈述的依据判定）, `score: float` | 低 | 最多尝试 2 次；仍失败则标记该 case 为 `eval_failed` |
| `build_rag_relevance_prompt` | `RelevanceScore` | `score: float`, `justification: str` | 低 | 同上 |
| `build_hallucination_check_prompt` | `HallucinationReport` | `unsupported_claims: list[str]`, `unsupported_ratio: float` | 低 | 同上 |
| `build_decision_trace_prompt` | `DecisionTraceScore` | `discrimination: int`（1-5）, `necessity: int`（1-5）, `priority: int`（1-5）, `evidence_completeness: int`（1-5） | 低 | 同上 |
| `build_response_quality_prompt` | `ResponseQualityScore` | `accuracy: int`（1-5）, `completeness: int`（1-5）, `safety: int`（1-5） | 低 | 同上 |
| `build_advice_completeness_prompt` | `AdviceCompletenessScore` | `medication_covered: bool`, `exam_covered: bool`, `risk_covered: bool`, `score: float` | 低 | 同上 |
| `build_patient_simulation_prompt` | 自由文本（无 Schema） | 模拟患者回答，自由文本输出 | — | 失败则该 E2E case 终止 |

---

## 9.4 不需要结构化输出的 LLM 调用

以下调用点输出为自然语言文本，直接作为面向患者/用户的回复内容，不施加 JSON Schema 约束。这些调用点**仍需按 §9.1 "实现风格约定"裸写 `try/except/finally` 埋点**——只是没有 `with_structured_output(Schema)` 这一步，`_attempts` / `_failures` / `_latency` 三个指标的 `schema` 标签固定填 `"free_text"`；`_retries` / `_fallbacks` / `_diagnose_reason` 按需上报：

| 调用点 | 输出形式 | 说明 |
|-------|---------|------|
| ⑥a `generate_followup` | 自然语言追问句 | 面向患者的口语化问题，不需要结构化 |
| ⑬ `format_response` | 自然语言最终回复 | 整合诊断与建议的患者可读回复 + 免责声明 |
| 4.2.4 `compact_context` | 压缩摘要文本 | 内部上下文压缩，仅供后续节点 prompt 拼装使用（当前未启用） |
| 评估层 `patient_simulation` | 模拟患者回答 | 自由文本角色扮演 |

---

## 9.5 全量 Pydantic Schema 定义

本节是所有 LLM 结构化输出 Schema 的**权威完整定义**，按 `src/agent/schemas/` 文件组织。§9.3 表格仅作快速查阅索引，字段细节以此处为准。

> **通用约定**：所有 Schema 均继承 `pydantic.BaseModel`，通过 `llm.with_structured_output(SchemaClass)` 约束 LLM 输出。以下代码块中省略 `from pydantic import BaseModel, Field` 等 import 语句。

---

##### 1. `info_collect.py` — 主诉提取输出

```python
# —— 子模型：被 InfoCollectOutput.present_illness_slots 引用 ——
class PresentIllnessSlots(BaseModel):
    """现病史结构化要素槽位（13 个维度），未提及的维度为 None/空列表"""
    onset_time:          str | None = Field(None, description="起病时间，如'3天前'")
    onset_mode:          str | None = Field(None, description="起病方式：急性/缓慢/隐匿")
    trigger:             str | None = Field(None, description="诱因：劳累/受凉/进食/无明显诱因")
    location:            str | None = Field(None, description="部位")
    nature:              str | None = Field(None, description="性质：刺痛/胀痛/绞痛/烧灼感")
    severity:            str | None = Field(None, description="程度：轻/中/重/VAS评分")
    duration_pattern:    str | None = Field(None, description="时间规律：持续性/间歇性/阵发性")
    aggravating:         list[str] = Field(default_factory=list, description="加重因素")
    relieving:           list[str] = Field(default_factory=list, description="缓解因素")
    associated_symptoms: list[str] = Field(default_factory=list, description="伴随症状（患者自述）")
    progression:         str | None = Field(None, description="病程演变：加重/减轻/稳定/波动")
    treatment_tried:     str | None = Field(None, description="诊疗经过：看过没、用过什么药")
    treatment_response:  str | None = Field(None, description="治疗反应：有效/无效/加重")

# —— 主模型：传给 llm.with_structured_output() ——
class InfoCollectOutput(BaseModel):
    """① info_collect Step 1 LLM 输出"""
    chief_complaint:      str = Field(..., description="主诉（主要症状+持续时间），如'腹痛3天'")
    present_illness:      str = Field(..., description="现病史自由文本（本次发病的详细展开）")
    present_illness_slots: PresentIllnessSlots = Field(..., description="现病史结构化槽位，与 present_illness 同步填充")
```

---

##### 2. `report_parser.py` — 检查报告解析输出

```python
# —— 子模型：被 ReportFindings.findings 引用 ——
class ReportFinding(BaseModel):
    """单份报告的结构化关键发现"""
    report_type:       str       = Field(..., description="报告类型：blood_routine / urine_routine / biochemistry / imaging / ecg / physical_exam / pathology / other")
    report_date:       str | None = Field(None, description="报告日期（YYYY-MM-DD），无法识别则为 None")
    abnormal_values:   list[str] = Field(default_factory=list, description="异常检验值，保留原始数值，如'WBC 12.3×10⁹/L↑'")
    impressions:       list[str] = Field(default_factory=list, description="诊断印象，如'右肺上叶磨玻璃结节'")
    positive_findings: list[str] = Field(default_factory=list, description="阳性发现（含异常值的临床解读，使用医学文献语言）")
    negative_findings: list[str] = Field(default_factory=list, description="阴性发现 / 已排除项，如'未见肝内胆管扩张'")

# —— 主模型：传给 llm.with_structured_output() ——
class ReportFindings(BaseModel):
    """①.5 / ⑨ 报告解析 LLM 输出"""
    findings: list[ReportFinding] = Field(default_factory=list, description="各份报告的结构化发现列表")
```

> **注**：`ReportFinding` 不含 `report_index` 字段——该字段由节点代码在写入 State `report_findings` 时根据 `exam_reports` 下标自动填充，不需要 LLM 输出。

---

##### 3. `ner.py` — 命名实体识别输出

```python
# —— 子模型：被 NERResult.entities 引用 ——
class NEREntity(BaseModel):
    """单个医学命名实体"""
    text:        str = Field(..., description="实体原文")
    entity_type: Literal["symptom", "disease", "drug", "anatomy"] = Field(..., description="实体类型")
    negation:    bool = Field(False, description="是否为否定表述，如'不头痛'")
    temporality: Literal["current", "past", "family"] = Field("current", description="时间属性：当前/既往/家族")
    value:       str | None = Field(None, description="量化值（如体温 38.5°C），无则 None")

# —— 主模型：传给 llm.with_structured_output() ——
class NERResult(BaseModel):
    """② build_query Step 1 NER LLM 输出"""
    entities: list[NEREntity] = Field(default_factory=list, description="识别到的医学实体列表")
```

---

##### 4. `entity_linking.py` — 已删除

EL 整层移除,`src/agent/schemas/entity_linking.py` 删,运行时不再有 `EntityLinkingMatch` 这个返回结构。详情见 §4.1.6.2 + EL_DESIGN_REVIEW §11。

---

##### 5. `query_construction.py` — Query 构建输出

```python
# —— 主模型:传给 llm.with_structured_output(),无子模型 ——
class QueryConstructionOutput(BaseModel):
    """② build_query Step 3 Query 构建 LLM 输出 — 仅 dense_query 一字段。

    sparse_queries 由 Step 2(state 多字段直采)确定性产出,LLM 不参与;
    曾把 sparse_queries 也作为 LLM 输出字段(为 schema 完整),但 LLM 看到 prompt
    里的"sparse 已定不要改"会合理省略输出,触发 schema 校验失败。改为 LLM 只承担
    dense_query 改写一职,避免 prompt/schema 内在冲突。
    """
    dense_query: str = Field(..., description="用于 Dense 检索的语义查询文本")
```

---

##### 6. `symptom_selection.py` — 追问症状选择输出

```python
# —— 子模型:被 SmartFollowupOutput.questions 引用 ——
class FollowupQuestion(BaseModel):
    """⑤ select_symptom 单条追问项。"""
    type: Literal["slot", "open"] = Field(...,
        description="slot=补全 13 维 HPI 空槽;open=开放式问'还有别的不舒服吗'")
    slot: str | None = Field(None,
        description="type=slot 时填,如 'trigger' / 'location' / 'nature' 等 13 维槽位名;type=open 时为 None")

# —— 子模型:被 SmartFollowupOutput.unaskable_symptoms / DiagnosisOutput.retained_unaskable 引用 ——
class UnaskableSymptom(BaseModel):
    """LLM 想知道但患者答不上的体征/指标(⑤ 粗筛 + ⑩ 精筛共用 schema)。

    ⑤ 出粗筛喂给 ⑩ Step 2 判 need_exam;⑩ Step 3 基于诊断结果挑出"仍需检查
    确认的"写回 state.unaskable_symptoms,⑧a 直接消费 description 作为检查建议来源。
    """
    description: str = Field(..., description="医生侧语言:想查什么 / 想知道什么体征,如'腹部 B 超提示有无胆囊壁增厚'")
    reason:      str = Field(..., description="为什么对鉴别诊断重要,如'关键鉴别胆囊炎 vs 胃炎'")

# —— 主模型:传给 llm.with_structured_output() ——
class SmartFollowupOutput(BaseModel):
    """⑤ select_symptom LLM 输出 — 1 次调用同时出 2 件事。

    LLM 输入 patient state(主诉 + 13 维 slots 空缺 + 已问症状),输出:
    - questions:追问项(slot 维度补全 / open 开放式),≤ MAX_FOLLOWUP_QUESTIONS,可为 0
    - unaskable_symptoms:想知道但患者答不上的体征粗筛(后续 ⑩ Step 3 会精筛覆盖)

    两个任务互斥:可问的进 questions,不可问的进 unaskable_symptoms,不重叠。
    questions 为空 → 信息已足,should_continue 路由跳诊断。
    """
    questions: list[FollowupQuestion] = Field(default_factory=list, max_length=5,
        description="本轮追问项列表(0-5 个);为空 = 信息已足,直接进诊断")
    unaskable_symptoms: list[UnaskableSymptom] = Field(default_factory=list, max_length=5,
        description="LLM 想知道但患者答不上的体征/指标(0-5 条粗筛);为空 = 没有需检查鉴别的项")
```

---

##### 7. `followup.py` — 追问回答解析输出

```python
# —— 主模型:传给 llm.with_structured_output() ——
class FollowupParseResult(BaseModel):
    """⑦ process_followup_answer LLM 输出。

    ⑤ 重设计后只产 slot / open 两类追问,⑦ 不再有"症状级 yes/no 回答分流"。
    - slot_fills: 维度级回填(对应 ⑤ 的 type=slot)
    - new_symptoms: 患者回答中提及的新症状(对应 ⑤ 的 type=open,或顺带补充),
      由 ⑦ 直接 append 到 confirmed_symptoms 供下轮 build_query 使用
    """
    slot_fills:   dict[str, str | list[str]] = Field(default_factory=dict, description="维度级回填,key=槽位名;value 类型与 PresentIllnessSlots 槽位一致")
    new_symptoms: list[str]                  = Field(default_factory=list, description="患者回答中提及的新症状(开放式追问的主要产物,也含顺带补充)")
```

---

##### 8. `diagnosis.py` — 诊断推理输出（1 步 LLM）

> 注：以下 Schema 也在 4.1.2 ⑩ 中内联展示供上下文阅读，此处为权威版本。⑩ 重设计:
> 3 步链 → 1 步(对齐 RAG 评测口径 `.eval/rag_eval/run_diagnose_eval.py`),旧的
> `EvidenceSheet` / `CandidateEvidence` / `HistoryFactor` / `SlotRelevance` /
> `ReportEvidence` / `DiagnosisRanking` 整体废弃。

```python
# —— 子模型：被 DiagnosisOutput.results 引用 ——
class RankedDisease(BaseModel):
    """单个候选疾病的诊断结果(字段对齐评测 CandidateDiagnosis + 生产新增 differentiation_type)"""
    disease:              str         = Field(..., description="疾病名;尽量精确到部位/分型(如 '右额颞急性硬膜外血肿' 而非 '颅内血肿');兜底场景固定为 '信息不足以支持可靠诊断'")
    probability:          float       = Field(..., ge=0.0, le=1.0, description="概率;兜底场景为 0.0")
    evidence:             list[str]   = Field(default_factory=list, description="3-5 条关键支持证据(可引用症状/报告/文献/图像)")
    differentiation:      str | None  = Field(None, description="与其他相似疾病的鉴别要点(可空)")
    differentiation_type: Literal["confirmed", "need_exam", "insufficient"] = Field(..., description="鉴别状态;top1 决定 router 走 ⑧ recommend_exam(need_exam)还是 ⑪ safety_gate(其他)")
    failure_reason:       str | None  = Field(None, description="系统级失败原因(非自然 insufficient)。取值示例:'followup_round_capped'(追问触顶)、'step_1_structured_output_failed: ValidationError: ...'(LLM 结构化输出失败)。None 表示 LLM 正常推理。该字段由节点代码在兜底路径中填充,不由 LLM 输出;供 ⑫ generate_advice 附加系统级提示、⑬ format_response 生成免责说明、rag_trace.error_info 审计追溯使用")

# —— 主模型：传给 llm.with_structured_output() ——
class DiagnosisOutput(BaseModel):
    """⑩ diagnose 1 步 LLM 输出 — 诊断结果 + 精筛 unaskable。

    retained_unaskable 是 LLM 基于当前诊断结果挑/改写的"仍需检查确认"的 unaskable
    列表（从输入的 ⑤ 粗筛版里筛 + 必要时改写描述），节点代码写回 state.unaskable_symptoms
    供 ⑧a recommend_exam 消费。LLM 判断不再需要的 → 不写进 retained_unaskable，自然丢弃。
    """
    results: list[RankedDisease] = Field(..., min_length=1,
                                         description="按 probability 降序排列的诊断结果列表;校验失败兜底为 [RankedDisease(disease='信息不足以支持可靠诊断', probability=0.0, ...)]")
    retained_unaskable: list[UnaskableSymptom] = Field(default_factory=list,
        description="基于诊断结果挑/改写后,仍需检查确认的 unaskable 列表（可为 ⑤ 粗筛版的子集或改写版）。confirmed/insufficient 路径下可为空（不会被消费）；need_exam 路径下应至少保留 1 条供 ⑧a 推荐检查")
```

---

##### 8.5. `recommend_exam.py` — 检查推荐输出

```python
class RecommendExamOutput(BaseModel):
    """⑧a recommend_exam LLM 结构化输出。
    spec §4.1.1 字段定义 `recommended_tests: list[str]` — 每项一个检查名,
    不允许把整段 free text 塞进 list 单元素破坏字段语义。"""
    tests: list[str] = Field(default_factory=list,
                             description="建议的检查项目清单(每项一个检查名,如'血常规'、'腹部 CT'、'胆囊超声');spec §4.1.2 ⑧ 期望 3-5 项;若所有所需检查患者都已上传报告则可为空")
    rationale: str = Field("",
                           description="整体推荐理由(为什么这几项 / 已有哪些可复用 / 哪个对鉴别最关键);面向患者的简短说明,2-3 句即可")
```

---

##### 9. `safety_gate.py` — 安全门控 LLM 兜底输出

```python
# —— 子模型：被 SafetyGateOutput.additional_risks 引用 ——
class SafetyRisk(BaseModel):
    """单项 LLM 识别的安全风险"""
    risk_type:      Literal["cross_allergy", "interaction", "dosage_adjustment"] = Field(..., description="风险类型")
    description:    str = Field(..., description="风险描述")
    severity:       Literal["high", "medium", "low"] = Field(..., description="严重程度")
    recommendation: str = Field(..., description="处置建议")

# —— 主模型：传给 llm.with_structured_output() ——
class SafetyGateOutput(BaseModel):
    """⑪ safety_gate LLM 兜底层输出"""
    additional_risks: list[SafetyRisk] = Field(default_factory=list,
                                                description="LLM 识别的规则层未覆盖的额外风险（交叉过敏、罕见相互作用等）")
```

---

##### 10. `advice.py` — 建议生成输出

```python
# —— 子模型：被 AdviceOutput.medications 引用 ——
class MedicationAdvice(BaseModel):
    """单项用药建议"""
    drug_name:  str        = Field(..., description="药品名称（通用名）")
    dosage:     str        = Field(..., description="剂量，如'0.1g'")
    frequency:  str        = Field(..., description="用药频次，如'每日3次'")
    duration:   str        = Field(..., description="疗程，如'7天'")
    notes:      str | None = Field(None, description="特殊注意事项（饭后服用、肝肾功能调整等）")

# —— 主模型：传给 llm.with_structured_output() ——
class AdviceOutput(BaseModel):
    """⑫ generate_advice LLM 输出"""
    medications:      list[MedicationAdvice] = Field(default_factory=list, description="用药建议列表（在 safety_constraints 约束内）")
    exam_suggestions: list[str]              = Field(default_factory=list, description="建议检查项目")
    risk_warnings:    list[str]              = Field(default_factory=list, description="风险提示与注意事项")
    urgent_flag:      bool                   = Field(False, description="是否高危情况（疑似心梗、脑卒中等），True 时强烈建议立即就医")
```

---

##### 11. `ingestion.py` — Chunk 增强输出

```python
# —— 主模型：传给 llm.with_structured_output()，无子模型 ——
class ChunkEnrichmentOutput(BaseModel):
    """3.1.3 enrichment LLM 输出 — 为原始 chunk 生成增强元数据"""
    title:                  str       = Field(..., description="chunk 标题（LLM 生成）")
    summary:                str       = Field(..., description="chunk 摘要（LLM 生成）")
    hypothetical_questions: list[str] = Field(default_factory=list, description="假设性问题（HyDE 反向生成，用于增强检索召回）")
```

---

##### 12. `evaluation.py` — 离线评估 LLM Judge 输出

```python
# --- RAG 忠实度评估 ---

# —— 子模型：被 FaithfulnessScore.claims 引用 ——
class ClaimJudgment(BaseModel):
    """单条陈述的依据判定"""
    claim:        str        = Field(..., description="从回复中提取的事实性陈述")
    supported:    bool       = Field(..., description="该陈述是否有 retrieved chunk 支撑")
    source_chunk: str | None = Field(None, description="支撑该陈述的 chunk 引用（无支撑则 None）")

# —— 主模型：传给 llm.with_structured_output() ——
class FaithfulnessScore(BaseModel):
    """RAG 忠实度评分"""
    claims: list[ClaimJudgment] = Field(default_factory=list, description="各陈述的依据判定")
    score:  float               = Field(..., ge=0.0, le=1.0, description="整体忠实度得分")

# --- RAG 相关性评估 ---

# —— 主模型：传给 llm.with_structured_output() ——
class RelevanceScore(BaseModel):
    """RAG 相关性评分"""
    score:         float = Field(..., ge=0.0, le=1.0, description="相关性得分")
    justification: str   = Field(..., description="评分理由")

# --- 幻觉检测 ---

# —— 主模型：传给 llm.with_structured_output() ——
class HallucinationReport(BaseModel):
    """幻觉检测报告"""
    unsupported_claims: list[str] = Field(default_factory=list, description="无依据的陈述列表")
    unsupported_ratio:  float     = Field(..., ge=0.0, le=1.0, description="无依据陈述占比")

# --- 诊断决策链评估 ---

# —— 主模型：传给 llm.with_structured_output() ——
class DecisionTraceScore(BaseModel):
    """诊断决策链质量评分"""
    discrimination:        int = Field(..., ge=1, le=5, description="鉴别诊断区分度（1-5）")
    necessity:             int = Field(..., ge=1, le=5, description="检查建议必要性（1-5）")
    priority:              int = Field(..., ge=1, le=5, description="优先级排序合理性（1-5）")
    evidence_completeness: int = Field(..., ge=1, le=5, description="证据完整性（1-5）")

# --- 回复质量评估 ---

# —— 主模型：传给 llm.with_structured_output() ——
class ResponseQualityScore(BaseModel):
    """最终回复质量评分"""
    accuracy:    int = Field(..., ge=1, le=5, description="准确性（1-5）")
    completeness: int = Field(..., ge=1, le=5, description="完整性（1-5）")
    safety:      int = Field(..., ge=1, le=5, description="安全性（1-5）")

# --- 建议完整性评估 ---

# —— 主模型：传给 llm.with_structured_output() ——
class AdviceCompletenessScore(BaseModel):
    """建议完整性评分"""
    medication_covered: bool  = Field(..., description="用药建议是否覆盖")
    exam_covered:       bool  = Field(..., description="检查建议是否覆盖")
    risk_covered:       bool  = Field(..., description="风险提示是否覆盖")
    score:              float = Field(..., ge=0.0, le=1.0, description="综合完整性得分")
```

## 9.6 审计埋点契约（`rag_trace` 写入规则）

**问题背景**：§5.2.3.1 定义了 `rag_trace` 表结构，但 Agent 节点（§4）和 G4 问诊接口（§8.3 阶段 G）都未明确"谁在何时往这张表写数据"。按章节读 spec 实现任务时会全盘跳过审计埋点，导致审计表建好但无数据写入，整个审计系统落空。本节统一定义 15 个字段的数据来源、写入主体、写入时机、错误字段生成规则，与 §9.1 "实现风格约定"保持一致（裸代码，不封装装饰器/helper）。

### 9.6.1 写入主体与时机

- **写入主体**：**API 层 G4 `POST /diagnose` endpoint**（`src/api/routes/diagnosis.py`）——Agent 内部节点**不直接写 rag_trace**，只负责把必要数据沉淀到 State（见 §9.6.2 数据来源对照表）。
- **写入时机**：每次 `graph.invoke(initial_state, config=config)` 返回后、`return response` 之前写一条。正常结束、兜底结束（`failure_reason` 非 None）、interrupt 恢复后自然结束，三种路径均需写入。
- **事务性**：`rag_trace` 写入独立于 Agent 主流程，写失败不阻塞响应，但必须 `logger.error()` 记录并触发 `structured_output_failure_total{node="rag_trace_write"}` 告警（复用 §9.1 指标体系的 Counter，或在 §H2 扩一个 `audit_write_failure_total`）。

### 9.6.2 15 字段数据来源对照表

| `rag_trace` 字段 | 类型 | 来源 | 具体表达式（Python 伪代码，`s = final_state`） |
|------------------|------|------|------------------------------------------------|
| `trace_id` | UUID, PK | 新生成 | `uuid.uuid4()` |
| `session_id` | UUID, FK → sessions | 请求上下文 | 从 FastAPI `Depends` / JWT 中拿 |
| `user_id` | UUID, FK → users | 请求上下文 | 从 FastAPI `Depends` / JWT 中拿 |
| `raw_query` | TEXT | State | `s["patient_input"]` |
| `intent_result` | JSONB | State 派生 | `{"chief_complaint": s["chief_complaint"], "confirmed_symptoms": s["confirmed_symptoms"], "denied_symptoms": s["denied_symptoms"]}`(意图识别并非独立节点,用 info_collect ① + build_query ② 的产物聚合;EL 移除后 standardized_entities 字段已删) |
| `retrieved_chunks` | JSONB | State | `s["candidate_chunks"]`（③ retrieve 写入的原始 Top-N 列表，含 RRF 分数） |
| `reranked_chunks` | JSONB | **新 State 字段** | `s["last_reranked_chunks"]`（⑩ Step 0 Cross-Encoder 精排后写入；Step 0 fallback 原序时即等于 `s["candidate_chunks"]`；兜底短路 Step -1 时为 `[]`） |
| `final_prompt` | TEXT | **新 State 字段** | `s["last_diagnose_prompt"]`（正常诊断 NULL；仅 ⑩ 失败兜底路径填值） |
| `llm_raw_output` | TEXT | **新 State 字段** | `s["last_diagnose_raw_output"]`（正常诊断 NULL；仅 ⑩ 失败兜底路径填值） |
| `final_response` | TEXT | State | `s["final_response"]` |
| `model_name` | VARCHAR(64) | config | `settings.llm.MODEL_NAME`（诊断节点用的 LLM;MVP 全流程共用单一模型,后期需要按节点分流时再扩展 settings.llm 子字段） |
| `token_usage` | JSONB | **新 State 字段** | `s["session_token_usage"]`（`RetryObserver.on_llm_end` 累加；初始全 0） |
| `latency_ms` | JSONB | **新 State 字段（需求和 total）** | `{**s["session_latency_ms"], "total": sum(s["session_latency_ms"].values())}` |
| `error_info` | JSONB | State 派生 | `_build_error_info(s["diagnosis_result"])`（见 §9.6.3 规则） |
| `created_at` | TIMESTAMPTZ | 数据库 | `DEFAULT now()`（SQLAlchemy `server_default=func.now()`） |

**说明**：
- "新 State 字段"共 5 个，定义见 §4.1.1（`last_reranked_chunks` / `session_token_usage` / `session_latency_ms` / `last_diagnose_prompt` / `last_diagnose_raw_output`），初始值见 §4.1.1a。
- `latency_ms` 中的 `total` 由 API 层现场求和，不需要节点额外维护。
- `token_usage` 的累加机制复用 §9.1 `RetryObserver` callback 的 `on_llm_end` 钩子，不为埋点另造一套。

### 9.6.3 `error_info` 填充规则

```python
def _build_error_info(diagnosis_result: list[dict]) -> dict | None:
    """从 diagnosis_result[0].failure_reason 派生 error_info。
    正常 LLM 推理结果 → 返回 None（入库为 NULL）。
    系统级失败 → 返回结构化 dict，供运维聚合。
    """
    if not diagnosis_result:
        return {"source": "diagnose", "failure_reason": "empty_diagnosis_result", "step": None}
    reason = diagnosis_result[0].get("failure_reason")
    if reason is None:
        return None
    # reason 取值来自 ⑩ diagnose：
    #   "followup_round_capped"
    #   "step_{1|2|3}_structured_output_failed: <ExcType>: <msg>"
    step = None
    if reason.startswith("step_") and "_structured_output_failed" in reason:
        try:
            step = int(reason.split("_")[1])
        except (IndexError, ValueError):
            step = None
    return {"source": "diagnose", "failure_reason": reason, "step": step}
```

### 9.6.4 与 Prometheus 指标的关系（两个独立系统）

| 维度 | `rag_trace`（§5.2.3.1） | Prometheus 指标（§9.1 + H2） |
|------|------------------------|------------------------------|
| 存储 | PostgreSQL，per-session 一行 | 时序数据库，聚合计数/直方图 |
| 用途 | **深度回溯**单次会话的完整链路 | **趋势分析**失败率、延迟分位数、QPS |
| 保留期 | 90 天（见 §5.2.3.5） | Prometheus 默认 15 天 |
| 查询方式 | SQL（按 trace_id / session_id / user_id） | PromQL（按 node / schema / exception_type） |
| 数据是否重复 | **否**。两者承载不同粒度（单次 vs 聚合） | **否**。Prometheus 不存 prompt / chunk 文本 |

**规则**：G4 同时写 `rag_trace`（DB）和 §9.1 定义的 6 个 Prometheus 指标（内存计数，由 `/metrics` 端点暴露）；两者数据不互相依赖，一方失败不影响另一方。

### 9.6.5 裸代码写入模板（严禁封装装饰器 / helper 类）

> **实现风格约定**（对齐 §9.1）：直接在 `POST /diagnose` 的视图函数内组装 dict 并调用 SQLAlchemy session 写入。**禁止**封装成 `@audit_rag_trace` 装饰器、`AuditWriter` 类、上下文管理器——写入主体唯一（G4），裸代码不会有复用问题，反而把数据来源摊在代码里一目了然。

```python
# src/api/routes/diagnosis.py 视图函数（裸代码样板）
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.agent.graph import compiled_graph          # LangGraph 编译产物
from src.agent.state import create_initial_state
from src.db.postgres.models import RagTrace
from src.db.postgres.session import get_session
from config.settings import settings

router = APIRouter()

@router.post("/diagnose")
async def diagnose(req: DiagnoseRequest,
                   current_user = Depends(get_current_user),
                   db: Session = Depends(get_session)):
    # 1. 构造初始 State 并调用 Agent graph
    initial_state = create_initial_state(
        patient_id=current_user.patient_id,
        patient_input=req.patient_input,
    )
    config = {
        "configurable": {"thread_id": f"session_{req.session_id}"},
        "callbacks": [retry_observer],               # §9.1，用于累加 token_usage
        "metadata": {"session_id": str(req.session_id)},
    }
    try:
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.error("graph_invoke_failed", exc_info=True)
        raise HTTPException(500, "诊断服务暂不可用，请稍后再试")

    # 2. 组装 rag_trace 记录（15 字段按 §9.6.2 对照表）
    s = final_state
    trace_row = RagTrace(
        trace_id=uuid.uuid4(),
        session_id=req.session_id,
        user_id=current_user.user_id,
        raw_query=s["patient_input"],
        intent_result={
            "chief_complaint": s["chief_complaint"],
            "confirmed_symptoms": s["confirmed_symptoms"],
            "denied_symptoms": s["denied_symptoms"],
            # EL 移除后 standardized_entities 字段已删
        },
        retrieved_chunks=s["candidate_chunks"],
        reranked_chunks=s["last_reranked_chunks"],
        final_prompt=s["last_diagnose_prompt"],      # 正常时 None → DB NULL
        llm_raw_output=s["last_diagnose_raw_output"],# 正常时 None → DB NULL
        final_response=s["final_response"],
        model_name=settings.llm.MODEL_NAME,
        token_usage=s["session_token_usage"],
        latency_ms={
            **s["session_latency_ms"],
            "total": sum(s["session_latency_ms"].values()),
        },
        error_info=_build_error_info(s["diagnosis_result"]),
    )

    # 3. 写入 DB（失败不阻塞响应，但要告警）
    try:
        db.add(trace_row)
        db.commit()
    except Exception:
        db.rollback()
        logger.error("rag_trace_write_failed", exc_info=True,
                     extra={"session_id": str(req.session_id)})
        # 不 raise：响应仍要返回给用户

    # 4. 返回响应
    return DiagnoseResponse.from_state(s)
```

### 9.6.6 与 §4 各节点的职责切分

| 角色 | 职责 | 不做什么 |
|------|------|---------|
| Agent 节点（①~⑬） | 把必要数据放进 State（如 ⑩ Step 0 写 `last_reranked_chunks`，⑩ 失败兜底写 `last_diagnose_prompt/raw_output`） | 不调 `INSERT INTO rag_trace` |
| `RetryObserver`（§9.1） | `on_llm_end` 累加 `session_token_usage`；`on_llm_start/on_llm_end` 测量并累加 `session_latency_ms["llm_call"]` | 不碰其他环节延迟（由各节点在自己代码内测完后写 State） |
| G4 endpoint | 从 final State 组装 `rag_trace` 记录并写 DB | 不算业务逻辑，不改 State |

## 9.7 运行时常量集中（`agent_limits`）

**问题背景**:代码层"硬性上限"与"阈值调优"类常量散落 §3 / §4,分章节实现各自任务时易写 magic number 或起不同键名,后期阈值调优需要改多处代码。本节列出此类常量的权威清单、定义位置、导入约定。

### 9.7.1 常量清单

| 常量名 | 初始值 | 用途 | 主要使用位置 |
|--------|--------|------|--------------|
| `MAX_FOLLOWUP_ROUNDS` | `8` | 追问轮次硬性兜底上限 | `should_continue`(§4.1.3.1)/ ⑩ Step -1(§4.1.2)|
| `MAX_EXAM_ROUNDS` | `3` | 检查循环硬性上限 | `diagnose_router`(§4.1.3.2)/ ⑧a `recommend_exam`(§4.1.2)|
| `MAX_FOLLOWUP_QUESTIONS` | `5` | 单轮追问问题条数上限(slot + open type 合计) | ⑤ `select_discriminative_symptom`(§4.1.2)|
| `RETRIEVE_TOP_N` | `200` | RRF 融合后 Top-N 截断(送入 ⑩ Step 0 Cross-Encoder) | ③ `retrieve`(§4.1.2)/ §3.2.2 |
| ~~`ASKABLE_GAIN_THRESHOLD`~~ | ~~`0.15`~~ | **已删除** — ⑤ 重设计为 1 LLM 直接选追问,信息增益机制废 | — |
| ~~`ENTITY_LINKING_TIER2_THRESHOLD`~~ | ~~`0.92`~~ | **已删除** — EL 整层移除,该阈值不再有意义。详见 §4.1.6.2 + EL_DESIGN_REVIEW §11 | — |
| `RERANKER_CUTOFF_LAYERS` | `None`（=全层不截断；模型 layerwise 完整深度，BGE-Reranker-v2-minicpm-layerwise 为 40 层） | Cross-Encoder layerwise early-exit 截断层数；`None` = 跑满全层 | ⑩ Step 0 / Reranker 客户端（§2.3，§3.2.3）|
| `RETRIEVE_PARENT_FIGURE_CAP` | `5` | Context 扩展规则 3:父块在 LLM context 里能带的同节图表数封顶（`chunk_type ∈ {table, figure}` 计数;按 `relative_chunk_index` 升序保留前 K 个） | ⑩ Step 0 后 / Context 扩展(§3.2.3)|
| `RRF_DENSE_WEIGHT_FACTOR` | `5` | RRF 加权融合:dense 路加权 `max(1, N_sparse/factor)`,sparse 各路等权 1 票。2026-05-17 RETRIEVAL_EVAL §4 评测确定 — sparse 多字段直采后 N_sparse=12~30,等权下 dense 被挤兑,N/5 后 D/S ≈ 1:3~1:4 | ③ retrieve fusion / §3.2.2 |

### 9.7.2 定义位置与类型

**位置**：`config/settings.py`，作为 Pydantic `BaseSettings`（v2 `pydantic_settings.BaseSettings`）的一个嵌套段 `agent_limits`。

```python
# config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AgentLimitsSettings(BaseSettings):
    """§9.7 运行时常量——硬性上限与阈值，支持 .env 覆盖"""
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    MAX_FOLLOWUP_ROUNDS:           int   = Field(8,    description="追问轮次硬性兜底上限")
    MAX_EXAM_ROUNDS:               int   = Field(3,    description="检查循环硬性上限")
    MAX_FOLLOWUP_QUESTIONS:        int   = Field(5,    description="单轮追问问题条数上限")
    RETRIEVE_TOP_N:                int   = Field(200,  description="RRF 融合后 Top-N 截断")
    RERANKER_CUTOFF_LAYERS:        int | None = Field(None, description="Cross-Encoder 提前退出层数，None=全层")
    RETRIEVE_PARENT_FIGURE_CAP:    int   = Field(5,    description="Context 扩展:父块在 LLM context 里能带的同节图表数封顶")
    RRF_DENSE_WEIGHT_FACTOR:       int   = Field(5,    description="RRF 加权融合:dense_weight = max(1, N_sparse/factor)")

class Settings(BaseSettings):
    # ... 其他段（llm / milvus / postgres / ...）
    agent_limits: AgentLimitsSettings = AgentLimitsSettings()

settings = Settings()  # 模块级单例
```

### 9.7.3 导入约定（业务代码使用方式）

```python
# src/agent/nodes/select_symptom.py（示例)
from config.settings import settings

def select_discriminative_symptom(state: MedicalState) -> dict:
    limits = settings.agent_limits
    # LLM 输出后,代码侧兜底截到 quota(LLM schema 已 max_length=5,这里再截一次)
    followup_questions = followup_questions[: limits.MAX_FOLLOWUP_QUESTIONS]
    return {"followup_questions": followup_questions, ...}
```

### 9.7.4 硬性规则

1. **业务代码禁止 hardcode**：`state["followup_round"] >= 8` 必须改成 `>= settings.agent_limits.MAX_FOLLOWUP_ROUNDS`，否则视为违规实现。
2. **阈值调优只改 .env**：通过 `AGENT_MAX_FOLLOWUP_ROUNDS=10` 覆盖，不改代码。
3. **常量必须集中在 `AgentLimitsSettings`**：新增此类运行时常量时先在本表加一行、再到 `config/settings.py` 加字段，再在业务代码里 import；**禁止**在业务模块里 `MAX_X = 8` 这种模块级私有常量（一旦多点实现就键名散乱）。
4. **与 §9.1 风格一致**：`settings` 对象也是模块级单例（对齐 `metrics.py` 的单例指标对象），裸代码直接 `import`，不封装装饰器/helper。

## 9.8 跨章节数据契约快速参考

**目的**：第 2 章定义的核心 schema 被第 3/4 章大量引用，按章节读 spec 实现任务时，需要一站式拿到字段清单。本节从权威章节复制关键 schema 的字段定义，方便 `09-contracts.md` 独立可读。

> **注**：本节是**快速参考**，字段权威定义以原章节（如 §2.4.6）为准；**新增字段或修改字段必须先改原章节，再同步到本节**。

### 9.8.1 `terms_collection` Schema 摘要（权威定义见 §2.4.6）

Milvus 术语向量库,**EL 移除后运行时不再被任何节点使用**(原 `build_query` ② Step 2 EL / `extract_symptoms` ④ Tier 2 / 3.2.1 alias 反查全部下线;详见 §4.1.6.2)。数据保留备用。

**集合字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `concept_id` | VARCHAR(64), PK | 标准概念 ID（如 ICD-10 `R10.4` / CMeSH `D010149` / PROJECT 自定义 ID） |
| `preferred_term` | VARCHAR(256) | 标准首选术语（如"腹痛"） |
| `entity_type` | VARCHAR(32) | 实体类型：`symptom` / `disease` / `drug` / `anatomy` |
| `alias` | VARCHAR(256) | 别名（单条记录；多别名 → 多行记录，通过 `concept_id` 关联） |
| `alias_embedding` | FLOAT_VECTOR(1024) | `alias` 的 Qwen3-Embedding-8B 向量（Cosine Similarity 检索） |
| `source_vocab` | VARCHAR(32) | 来源词典：`ICD-10-CN` / `CMeSH` / `PROJECT`（自建医学术语） |
| `category` | VARCHAR(64) \| NULL | 分类标签（如症状的系统归属 `digestive` / `respiratory`） |
| `icd10` | VARCHAR(16) \| NULL | 关联 ICD-10 代码（entity_type=disease 时必填，symptom 时可选） |

**索引配置**：
- Dense 向量索引：`alias_embedding` → `HNSW` (M=16, efConstruction=256, metric=COSINE)
- 标量索引：`concept_id`（PK 自带）、`entity_type`、`source_vocab`

**典型使用模式**（实现 F3 `build_query` / F5 `extract_symptoms` 时查看）：
- ~~**Entity Linking Top-5 查询**~~:**EL 移除后运行时已不再调用**(原对患者口语 `raw_text` 做 Qwen3-Embedding-8B 编码 → `alias_embedding` Top-5 ANN → 阈值过滤;`ENTITY_LINKING_TIER2_THRESHOLD` 常量同步删除,见 §9.7.1)。
- **同义词扩展**：以命中的 `concept_id` 为主键 → 查该 `concept_id` 下所有 `alias` 记录 → 合并为词袋（Sparse 路 BM25 用）。

### 9.8.2 扩展约定

将来若有其他跨章节高频引用的 schema（如 `docs_collection` / `chunks` 表结构），追加到本节对应小节即可（如 §9.8.3 / §9.8.4）。新增时遵循同一原则：**只复制字段清单 + 索引 + 典型使用模式**，不复制权威章节的完整设计论证。
