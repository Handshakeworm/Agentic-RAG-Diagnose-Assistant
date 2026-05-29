# 后端并发与扩展性评估（Concurrency & Scalability Review）

> **范围**：本文聚焦「非 AI/RAG 的纯后端基础设施」——FastAPI / uvicorn、PostgreSQL、Redis、Milvus 连接、中间件、进程与 GPU 资源模型——评估其**高并发承载力（high-concurrency capacity）**与**横向扩展能力（horizontal scalability）**。
>
> **方法**：8 维度多 agent 调研 workflow（26 个 agent），每个维度先由 finder agent 读真实代码出结论，再对每条 critical/high 结论做**对抗式验证（adversarial verification）**——验证 agent 被要求"尽量推翻原结论"，只有代码无歧义支持才确认。
>
> **重要免责**：文中所有延迟/吞吐数字均为**数量级估算（order-of-magnitude estimate）**,基于模型规模、batch、调用次数推算。仓库内目前**没有 query-time 压测基准**（[embedding_model.py](../src/models/embedding_model.py) docstring 只有 ingest-time 数据）。要拍板容量必须实测。
>
> 评估日期：2026-05-29 · 对应代码：`main` @ f1ee25d1

---

## TL;DR（一句话结论）

**不能扛 Web 级高并发,而且这一半是「有意为之」、一半是「真有 bug」。** 系统是**单进程 + 单 GPU + 单副本（single-process / single-GPU / single-replica）**的有状态（stateful）架构,现实并发天花板约 **5~20 个同时问诊（concurrent consultations）**,再多就排队、超时。关键认知:**"纯后端"（FastAPI/Redis/PG）与 AI 负载共享同一个事件循环 + 同一个线程池,无法单独评估**——这是单体（monolith）架构的本质特征。

对当前定位（单机、单 GPU 的"完成初诊" demo/作品集）而言,**单进程是正确取舍**（GPU 显存 16GB 决定）。但下文第 3 节那几个 bug **与定位无关,该修**。

---

## 1. 先纠正一个最常见的误判（the async/sync trap）

直觉推理:`POST /diagnose` 是 `async def`（[diagnosis.py:250](../src/api/routes/diagnosis.py#L250)）,而所有 Agent 节点都是同步 `def` + 同步 `.invoke()`/`.encode()`,**所以同步阻塞调用会卡死事件循环（event-loop blocking）,一个慢请求拖垮所有人**。

**这是错的。** 验证 agent 读了 LangGraph 源码确认:`add_node` 把每个同步 `def` 节点用 `run_in_executor` 包一层,**丢进默认线程池（default threadpool）执行**（`langgraph/_internal/_runnable.py:520-522` → `langchain_core/runnables/config.py:665-671`）。所以节点里的 LLM 调用、GPU 推理、Milvus 查询都跑在**工作线程**上,**不**阻塞事件循环。

> 工程交流里值得记住的表达:
> - "看起来会阻塞,但实际被 offload 到线程池了" → *"it looks blocking but it's offloaded to a threadpool via `run_in_executor`"*
> - 真正致命的是 **"on the event loop"**——在事件循环线程上跑的阻塞调用（见第 3.2 节）。

这条纠正是本次审计中最关键的一步:一个 finder agent 把"embedding 阻塞事件循环"判为 **critical**,验证 agent **推翻**为 medium。

---

## 2. 真正的瓶颈：一层套一层的串行资源（the serialization stack）

并发能力不是被"事件循环"卡住,而是被一串**串行化资源（serialized resources）**逐层收窄。从外到内:

| 层 | 限制 | 实际容量 | 中英术语 |
|---|---|---|---|
| ① 进程/Worker | `uvicorn --workers 1`（[Dockerfile.api](../infra/docker/Dockerfile.api)） | **1 个进程** | single worker process |
| ② 线程池 | 默认 `min(32, CPU+4)` | 16 核 → **~20 线程**;2 vCPU 容器 → **仅 6** | default threadpool / executor saturation |
| ③ **GPU** | 单卡,Embedding 9.3G + Reranker 2.6G,无锁 | **同一时刻只能 1 个推理** | single-GPU serialization point |
| ④ LLM 厂商配额 | DashScope/DeepSeek 账号级 QPS/TPM | 无客户端限流,撞墙即 429 | account-level rate limit (QPS/TPM) |
| ⑤ PG 连接池 | `pool_size=5 + overflow=10`（[connection.py:21-29](../src/db/postgres/connection.py#L21-L29)） | **15 条连接** | connection pool exhaustion |
| ⑥ 检查点存储 | `InMemorySaver` 在进程内存（[graph.py:176](../src/agent/graph.py#L176)） | **锁死 = 1 副本** | process-local checkpointer |

**木桶效应（the binding constraint）**:GPU（③）和单进程（①⑥）是最短的板。即便把所有异步问题都修好,GPU 单卡"一次只能算一个"就把吞吐封死在**每秒不到 1 个问诊**（开 reranker 跑 200 候选时）。

### 量化天花板（estimated ceiling）

- **线程池层**:~20 个节点可同时在飞,但单个 ⑩ diagnose 节点 LLM 超时上限 300s——长占线程。
- **现实稳态**:**~5~20 个并发问诊**后延迟堆积（latency pile-up）。失败模式是**线程池饱和 + PG 池 checkout 超时**,不是经典事件循环饿死。
- **纯 GPU 吞吐**:reranker 关（当前默认）→ embedding 约束,~7~20 encode/s;reranker 开跑 ~200 对 → **塌到 ~0.2~0.5 req/s**。
- **PG 池**:`get_db` 全程占连接,**~7~14 个并发问诊**耗尽 15 条池,第 16 个阻塞 30s 后报 500。
- **横向扩展**:`InMemorySaver` 进程内存 → **恰好 1 副本,无法加机器**。

---

## 3. 纯后端层真正存在的 bug（按优先级，含验证结论）

> 标注 ✅=验证确认 / ⚠️=验证后降级。每条附 `file:line` 与修复方案。

### 🔴 3.1 内存泄漏 → 必然 OOM（memory leak · ✅ confirmed）
`InMemorySaver` 把**每一个**中断会话（interrupted session）的完整 `MedicalState`（含 ~200 检索 chunk）永久留在进程内存,**从不清理（no eviction）**。问诊天然在入口就中断一次,用户半路弃用（abandonment）是常态。估算每会话 30~150KB,几万废弃会话吃光几 GB 堆,把还扛着 ~11.6GB 模型的进程 OOM。
**修复**:终态/出错时调 `checkpointer.delete_thread(...)`（[diagnosis.py:484](../src/api/routes/diagnosis.py#L484) 附近）+ 加 TTL 清扫。

### 🔴 3.2 同步 DB 写在事件循环上（the only real event-loop blocking · ✅ confirmed）
异步 `/diagnose` 里直接调同步 SQLAlchemy:[diagnosis.py:257](../src/api/routes/diagnosis.py#L257)、[269-272](../src/api/routes/diagnosis.py#L269-L272)、生成器里的 [473-475](../src/api/routes/diagnosis.py#L473-L475)。这些**没有** offload,直接冻结单个事件循环——写大 JSONB（`rag_trace`）再叠加 `pool_pre_ping` 额外往返,会让**所有**并发 SSE 流卡顿（jitter）。
**修复**:`await run_in_threadpool(...)` 包起来,或换异步引擎（`create_async_engine` + `AsyncSession`）。

### 🔴 3.3 单进程单副本 = 零冗余（no horizontal absorption · ✅ confirmed）
`workers=1` + 单 nginx upstream（[nginx.conf](../infra/docker/nginx.conf) keepalive 32）。任何循环停顿、线程池耗尽、单个长节点都拖垮 100% 流量;进程重启丢所有 in-flight 中断会话(代码已在 [diagnosis.py:21-23](../src/api/routes/diagnosis.py#L21-L23) TODO 标注)。600s 读超时让卡死的流堆积、占满 nginx keepalive 槽。
**修复（架构级,见第 5 节）**:GPU 模型拆独立推理服务 → API 层可 `workers>1`;`InMemorySaver` → `PostgresSaver`。

### 🟠 3.4 重试风暴（retry storm · ✅ confirmed）
OpenAI SDK 自带 `max_retries=2` **嵌套**在 `with_retry(stop_after_attempt=3)` 里 = 单次失败调用最多 **9 次往返**。一次问诊 8~15 个调用点,N 并发撞 429 放大成 N×9,**自我加剧（self-amplifying）**恶化它本想应对的限流。
**修复**:`get_llm()`（[llm_client.py:27-36](../src/models/llm_client.py#L27-L36)）里设 `max_retries=0`,只保留一层重试 + 带 jitter 的退避。

### 🟠 3.5 无客户端限流（no client-side rate limiting · ✅ confirmed）
对 LLM 厂商没有任何 QPS/TPM 节流,高并发盲撞账号配额,撞墙即喂给重试风暴。
**修复**:加 langchain `InMemoryRateLimiter`,按账号配额设 `requests_per_second`（DeepSeek / DashScope 各一个）。

### 🟠 3.6 GPU 无锁（unsynchronized GPU access · ⚠️ critical→medium）
Embedding/Reranker 单例**无锁**（[embedding_model.py](../src/models/embedding_model.py)）。今天因 reranker executor 是 `max_workers=1` + embedding 路径实际被串行才没炸;一旦有人"加线程提并发"就会 CUDA OOM（VRAM 已用 ~11.9G/16G,几无余量）。
**修复**:GPU 入口包进程级 `threading.Lock`（GPU 本就一次一个,串行化零成本）+ Prometheus gauge 观测。

### 🟡 3.7 Redis 限流在事件循环上 + SSE 缓冲（✅ confirmed / ⚠️ SSE buffering high→low）
限流中间件是 `BaseHTTPMiddleware`,在 `async dispatch` 里调**同步** Redis `is_allowed`（[rate_limiter.py:138-145](../src/api/middleware/rate_limiter.py#L138-L145)）——每个请求都在单循环线程上阻塞一次 Redis 往返。本地 Redis 影响小（亚毫秒）,但 Redis 变慢时吞吐塌到 ~1/RTT。`BaseHTTPMiddleware` 还会**缓冲流式响应（buffers streaming response）**,是 SSE 已知反模式。
**修复**:改写成纯 ASGI 中间件 + `redis.asyncio`;或先用 `await run_in_threadpool(...)` 包同步 backend。

> 👍 正面:[health.py](../src/api/routes/health.py) 已正确用 `run_in_executor` offload 探针——团队懂这个模式,只是没用在 `/diagnose` 热路径。LLM 客户端也正确**单例复用**（httpx 连接池复用,[llm_client.py:27](../src/models/llm_client.py#L27)）。

---

## 4. `get_db` 的连接浪费（connection held for whole stream · ✅ confirmed）

`get_db` 依赖让连接**在整个 SSE 流期间被占住**（[auth.py:48-63](../src/api/routes/auth.py#L48-L63)），哪怕真正 DB 操作只有几毫秒。一次问诊 LLM 阶段 2~3 分钟里那条连接全程闲置占用 → **~7~14 并发耗尽 15 条池**,第 16 个 checkout 阻塞 `pool_timeout`（未设 = SQLAlchemy 默认 30s）后报 500。更糟:单个问诊轮可能**同时占两条连接**（`get_db` + 节点内 `session_scope`）。
**修复**:`/diagnose` 别用 `Depends(get_db)`,改成只在两个真正读写点用短生命周期 `with session_scope()`,连接只持有毫秒级。

---

## 5. 横向扩展路线图（scaling roadmap）

分两类,正好对应"纯后端 vs AI 耦合":

### A. 纯后端就能修（不动 AI,先做）
1. **异步化 DB**:`create_async_engine` + `AsyncSession`,消灭 3.2 的事件循环阻塞。
2. **`InMemorySaver` → `PostgresSaver`**（依赖已在,差一张 `langgraph_checkpoints` 表 + Alembic 迁移）——一步**同时**解决 3.1 内存泄漏、重启丢会话、和"只能 1 副本"。
3. **中间件改纯 ASGI**;Redis 换 `redis.asyncio`。
4. **准入控制（admission control）**:信号量（semaphore）+ 队列,满了**快速失败 429**,而不是无声堆积。

### B. 必须拆架构才能水平扩展
- **GPU 模型拆独立推理服务（separate inference service）** → API 层就能 `workers>1` 甚至多副本（replicas）,前面挂负载均衡（load balancer）。
- 配 A 步的共享 PostgresSaver,任意副本都能 resume 任意会话,彻底解开"单进程锁死"。

做完 A+B,API 层（"纯后端"）才真正具备**无状态横向扩展（stateless horizontal scaling）**;GPU 服务单独按卡数扩。

---

## 6. 术语小抄（中英对照，面试/交流用）

| 中文 | English | 怎么用 |
|---|---|---|
| 事件循环阻塞 | event-loop blocking | "A sync call in an `async def` handler blocks the event loop" |
| 线程池饱和 | threadpool saturation / executor starvation | "sync nodes are offloaded but the threadpool saturates at ~20" |
| 串行化资源 | serialization point / serialized resource | "the single GPU is a hard serialization point" |
| 吞吐 / 延迟 | throughput / latency (p99 tail latency) | "throughput is GPU-bound, tail latency degrades under load" |
| 连接池耗尽 | connection pool exhaustion | "the 15-conn pool exhausts, checkout blocks then times out" |
| 准入控制 / 背压 | admission control / backpressure | "add a semaphore for admission control, shed load with 429" |
| 重试风暴 | retry storm (self-amplifying) | "nested retries cause a retry storm under 429s" |
| 水平/垂直扩展 | horizontal / vertical scaling | "InMemorySaver blocks horizontal scaling — pinned to 1 replica" |
| 有状态/无状态 | stateful / stateless | "make the API tier stateless via a shared checkpointer" |
| 优雅降级 / fail-open | graceful degradation / fail-open | "the rate limiter fails open when Redis is down" |
| 单点故障 | SPOF (single point of failure) | "PG/Milvus/Redis are all single non-replicated nodes" |

---

## 附录 A：8 维度完整 findings（含验证判定）

> 格式:`[原始严重度](验证判定→修正严重度) 标题`。无括号 = 该条未做对抗式验证（仅对 critical/high 验证）。

### A.1 Web 服务器 & 事件循环模型
- `[high](partially_correct→high)` 所有 LangGraph 节点是同步 `def`,被 offload 到**共享**默认线程池——真实并发上限 ~`min(32,cpu+4)`,与所有同步路由 handler 共用
- `[high](confirmed→high)` 同步 SQLAlchemy 写直接跑在异步 `/diagnose` 的事件循环上——卡顿所有并发 SSE 流
- `[high](confirmed→high)` `workers=1` 单进程 + 单 nginx upstream = 零冗余,一次循环停顿降级 100% 流量
- `[medium]` `RateLimitMiddleware` 是 `BaseHTTPMiddleware`,每请求在事件循环上调同步 backend
- `[low]` Reranker 每次调用新建 `ThreadPoolExecutor(max_workers=1)` 做超时,嵌套在已 offload 的 diagnose 节点里
- `[info]` GPU embedding/reranker 单例是真·天然串行资源,被所有并发问诊共享

### A.2 LangGraph 节点执行（同步 vs 异步）
- `[info]` 同步节点被 LangGraph 线程池 offload,**不**在事件循环上跑（纠正最坏假设）
- `[high](confirmed→high)` 默认循环线程池（max_workers≈20）是图执行的真实并发天花板
- `[high](confirmed→high)` GPU 模型单例（embedding+reranker）从工作线程调用且**无锁**——并发 GPU 争用
- `[medium]` Reranker 每次 ⑩ diagnose 都新建嵌套线程池
- `[medium]` 单次 `/diagnose` 扇出 ~6~10 次串行 LLM 往返 + GPU + Milvus,全压一个厂商

### A.3 PostgreSQL 连接池
- `[high](confirmed→high)` `get_db` 在整个 SSE 流期间占住连接,而非仅 DB 操作期间
- `[medium]` 单 uvicorn 进程使 15 连接成为**全局**上限（无横向余量）
- `[medium]` 单个问诊轮可能同时占两条连接（`get_db` + 节点内 `session_scope`）
- `[medium]` `pool_timeout` 未设 → 阻塞的 checkout 等默认 30s 才报错
- `[low]` 异步 handler/生成器里有直接同步 DB 调用跑在事件循环上
- `[info]` `pool_pre_ping=True` 每次 checkout 加一往返（正确,但 metrics gauge 未接）

### A.4 GPU 模型争用
- `[critical](refuted→medium)` ~~node ③ 同步 `.encode()` 阻塞事件循环~~ → **推翻**:被线程池 offload
- `[high](confirmed→medium)` GPU 推理无应用级锁——同模块并发前向传播未同步
- `[high](confirmed→medium)` 单 GPU 是硬串行点,Embedding(~9.3G)+Reranker(~2.6G) 一进程共享
- `[medium]` Reranker 正确 offload 但硬串行在 `max_workers=1`,且其超时无法真正中断 GPU
- `[low]` ingest-time 与 query-time embedding 共享同一 GPU 单例,无协调
- `[info]` 注:`reranker_model.py` 为空,真实 reranker 单例在 [reranker.py](../src/rag/retrieval/reranker.py)

### A.5 LLM 客户端（DashScope/DeepSeek）
- `[high](confirmed→high)` 隐藏双层重试:SDK `max_retries=2` 套 `with_retry(3)` = 单次失败最多 9 往返
- `[high](confirmed→high)` 无客户端 QPS/TPM 限流——账号配额是真实上限且盲撞
- `[medium]` 同步 `chain.invoke()` 跑在与同步 PG I/O 共享的 anyio 线程池,长 LLM 调用全程占线程
- `[medium]` 有单次超时但无单问诊全局 deadline;vision diagnose 可占线程 300s×重试
- `[info]` LLM 客户端正确单例复用（httpx 连接池复用）——正面

### A.6 Redis & 限流
- `[high](confirmed→high)` 同步 Redis 限流检查每请求在事件循环上跑
- `[high](partially_correct→low)` `BaseHTTPMiddleware` 缓冲响应——削弱/破坏 SSE 流
- `[low]` Redis 重启后首请求两次往返（SCRIPT LOAD+EVALSHA）串行在循环上
- `[medium]` 限流是全局共享（所有路由一个限额）,默认 30/min,对慢 SSE 端点偏粗
- `[low]` JWT 解码在限流 key 提取里同步跑在事件循环上（每请求 CPU 活）
- `[info]` 配置缓存 fail-open 设计正确,当前不在任何请求热路径

### A.7 Agent 状态 & 检查点 & 会话生命周期
- `[high](confirmed→high)` `InMemorySaver` 无界且从不清理——废弃中断会话泄漏完整 `MedicalState` 到 OOM
- `[high](confirmed→high)` 进程内检查点是硬横向扩展阻断——无法加第二副本
- `[medium]` 进程重启静默丢所有 in-flight 中断会话
- `[medium]` `MedicalState` 携带重 chunk/文本载荷进检查点,膨胀单会话内存
- `[info]` 单图单例上并发不同 `thread_id` 是正确安全的——无共享状态污染

### A.8 Milvus / 向量检索 & 数据层扩展
- `[info]` `src/db/milvus/connection.py` 是 0 字节空文件;连接态在 pymilvus 进程级全局单例
- `[high](confirmed→high)` 每次 retrieve 的 Milvus 往返严格串行:1 dense + N sparse(≈12-30),每次前置 `coll.load()`,无批/并行
- `[high](partially_correct→low)` GPU embedding 单例无 per-call 锁;并发 retrieve 在一个 CUDA context 上争用
- `[high](partially_correct→medium)` 所有同步节点体共享一个 ~20 线程默认池,无自定义 executor
- `[medium]` 单节点 Milvus standalone + 单 PG + 单 Redis + 单 GPU + `workers=1`:整个数据/计算层都是非冗余单点
- `[low]` `terms_collection` 向量搜索（实体链接 Tier 2）在同一共享 channel/线程池上再加串行 RPC

---

## 附录 B：优化优先级速查（建议落地顺序）

| 优先级 | 项 | 类型 | 改动量 | 收益 |
|---|---|---|---|---|
| P0 | 3.1 会话 `delete_thread` + TTL | 防 OOM | 小 | 防进程崩 |
| P0 | 3.2 DB 写 offload / 异步引擎 | 消除循环阻塞 | 中 | 所有 SSE 流不再卡顿 |
| P1 | 3.4 `max_retries=0` 去重试嵌套 | 防重试风暴 | 极小 | 高并发不自爆 |
| P1 | 4 `get_db` 改短生命周期 | 释放连接 | 小 | 池容量翻倍 |
| P1 | 3.5 客户端 LLM 限流 | 优雅降级 | 小 | 不盲撞配额 |
| P2 | 3.6 GPU 加锁 + gauge | 防 OOM race | 小 | 为加线程铺路 |
| P2 | 3.7 中间件改 ASGI + 异步 Redis | SSE 不缓冲 | 中 | 流式体验 |
| P3 | 5.B GPU 拆服务 + PostgresSaver | 横向扩展 | 大 | 真正多副本 |
