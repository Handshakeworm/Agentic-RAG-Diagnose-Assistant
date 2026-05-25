"""src/common/metrics.py — 可观测性原料(DEV_SPEC §9.1 + §4.2.7 + H2)。

**只暴露**(模块级单例 + RetryObserver,严禁封装装饰器 / helper / 上下文管理器):

- §9.1 结构化输出健康度(6 个,业务节点必带):
  `_attempts` / `_retries` / `_failures` / `_fallbacks` / `_latency` / `_diagnose_reason`
- §4.2.7 上下文 / 会话级(4 个,Agent 调 LLM 前 / 会话结束时 .observe()):
  `context_tokens_per_llm_call` / `context_structured_fields_size` /
  `context_messages_count` / `context_loop_iterations`
- 依赖层(H2,SDK 原生路径调用前 .observe()):
  `db_query_latency_seconds`(SQLAlchemy)/ `redis_command_latency_seconds`(redis-py)/
  `milvus_rpc_latency_seconds`(pymilvus)
- `RetryObserver`(LangChain `BaseCallbackHandler` 子类)+ `retry_observer` 单例

各 LLM 调用点按 §9.1 模板裸写 try/except/finally,主动 import 这里的指标做
`.labels(...).inc()` / `.observe()`,并把 `retry_observer` 通过 `config={"callbacks": [retry_observer]}`
传给 `chain.invoke(...)` —— `with_retry` 内部的重试事件由 callback 捕获,业务代码看不到。

20+ 调用点合计约 300 行重复样板,这是"零抽象"的明确代价,换取异常作用域清晰、
重试可观察、与 with_retry 内部行为不打架。
"""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from prometheus_client import Counter, Gauge, Histogram


# ────────────────────────────────────────────────────────────────────────────
# 6 个指标(spec §9.1 表格 + §4.2.7 结构化输出健康度表)
# ────────────────────────────────────────────────────────────────────────────


_attempts = Counter(
    "structured_output_attempt_total",
    "LLM 结构化输出调用尝试总数(进入 try 前 .inc())",
    ["node", "schema"],
)

_retries = Counter(
    "structured_output_retry_total",
    "LLM 结构化输出 with_retry 内部重试次数(由 RetryObserver.on_retry 捕获)",
    ["node", "schema"],
)

_failures = Counter(
    "structured_output_failure_total",
    "LLM 结构化输出尝试耗尽后仍失败的总数(except 分支 .inc())",
    ["node", "schema", "exception_type"],
)

_fallbacks = Counter(
    "structured_output_fallback_triggered_total",
    "LLM 调用失败后走兜底路径的总数(执行兜底前 .inc())",
    ["node", "fallback_type"],
)

_latency = Histogram(
    "structured_output_latency_seconds",
    "LLM 结构化输出调用端到端耗时(finally 块用 time.perf_counter() 差值 .observe())",
    ["node", "schema"],
    # 2026-05-25:默认桶最大 10s,LLM 调用真实范围 1-180s(flash 1-15s / pro 5-30s /
    # vision 30-180s),所有 > 10s 样本都被压平显示成"10s"。改自定义桶覆盖真实分布。
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0),
)

_diagnose_reason = Counter(
    "diagnose_failure_reason_total",
    "⑩ diagnose 节点失败原因分桶(按 failure_reason 写入时 .inc())",
    ["reason_kind"],
)


# ────────────────────────────────────────────────────────────────────────────
# 上下文 / 会话级(spec §4.2.7 表 1,4 个)
# ────────────────────────────────────────────────────────────────────────────


context_tokens_per_llm_call = Histogram(
    "context_tokens_per_llm_call",
    "每次 LLM 调用实际传入的 token 数(本地用 tiktoken 估算 prompt token,调 LLM 前 .observe())",
    ["node", "model"],
    buckets=(100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000),
)

context_structured_fields_size = Gauge(
    "context_structured_fields_size",
    "MedicalState 中结构化字段总体积(字符数),会话结束时或检查点 .set() 一次",
    ["session_id"],
)

context_messages_count = Histogram(
    "context_messages_count",
    "会话结束时 messages 列表长度",
    ["session_id"],
    buckets=(1, 3, 5, 8, 12, 18, 25, 40, 60, 100),
)

context_loop_iterations = Histogram(
    "context_loop_iterations",
    "追问 / 检查循环实际执行次数(spec §9.7 MAX_FOLLOWUP_ROUNDS=8 / MAX_EXAM_ROUNDS=3 是上限)",
    ["loop_type"],  # "followup" | "exam"
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 10),
)


# ────────────────────────────────────────────────────────────────────────────
# 依赖层(spec §5.2.1 ③ + H2,SDK 原生指标包装)
# ────────────────────────────────────────────────────────────────────────────


db_query_latency_seconds = Histogram(
    "db_query_latency_seconds",
    "PG 查询端到端耗时(SQLAlchemy before/after_cursor_execute 事件)",
    ["operation"],  # SELECT / INSERT / UPDATE / DELETE / OTHER
)

db_pool_checkedout = Gauge(
    "db_pool_checkedout",
    "PG 连接池正在使用的连接数(SQLAlchemy engine.pool.checkedout())",
)

redis_command_latency_seconds = Histogram(
    "redis_command_latency_seconds",
    "Redis 命令耗时(各调用点用 time.perf_counter 包一次性能开销 ~微秒)",
    ["command"],  # GET / SETEX / DEL / EVALSHA / PING / OTHER
)

milvus_rpc_latency_seconds = Histogram(
    "milvus_rpc_latency_seconds",
    "Milvus RPC 耗时(connection wrapper 在 search/query/insert/upsert 入口 .observe())",
    ["collection", "operation"],
)

milvus_rpc_errors_total = Counter(
    "milvus_rpc_errors_total",
    "Milvus RPC 错误计数(异常分类)",
    ["collection", "operation", "error_code"],
)


# ────────────────────────────────────────────────────────────────────────────
# RetryObserver — LangChain BaseCallbackHandler 子类,捕获 with_retry 重试事件
# ────────────────────────────────────────────────────────────────────────────


class RetryObserver(BaseCallbackHandler):
    """`with_retry` 内部的重试发生在 LangChain Runnable 内部,业务代码的
    try/except 看不到,所以用 callback 捕获(spec §9.1 注:这是 LangChain 框架
    原生扩展点,不是项目自建封装层)。

    业务侧用法(spec §9.1 模板):
        chain.invoke(
            prompt,
            config={"callbacks": [retry_observer], "metadata": {"node": ..., "schema": ...}},
        )
    """

    def on_retry(self, retry_state, *, metadata=None, **kwargs):
        md = metadata or {}
        _retries.labels(
            node=md.get("node", "unknown"),
            schema=md.get("schema", "unknown"),
        ).inc()


retry_observer = RetryObserver()
