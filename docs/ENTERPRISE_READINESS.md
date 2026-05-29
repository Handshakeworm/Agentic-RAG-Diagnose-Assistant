# 企业级成熟度路线图（Enterprise-Grade Readiness Roadmap）

> **用法**：这是一份"一块一块啃"的清单。每块结构统一——**是什么 / 真正的难点 / 你的现状 / 可勾选子任务 / 面试钩子 / 术语**。子任务用 `- [ ]` 勾,啃完打勾。
>
> **筛选原则（重要）**：只收录**技术含金量高 + 个人项目能加分**的维度。刻意**剔除"麻烦但不难"(tedious, not hard)**的体力活——合规/PHI、密钥托管、基础 RBAC 这些本质是流程,技术深度低,个人项目做了也难证明硬实力(见末尾"刻意跳过"一节)。
>
> **配套文档**:并发部分已单独成文 → [CONCURRENCY_REVIEW.md](CONCURRENCY_REVIEW.md)。
>
> 更新日期:2026-05-29 · 对应代码:`main`

---

## 0. 总览：两桶分法 + 优先级

一个面试就能讲的视角:把企业级关注点分两桶——

> **桶一 · 通用后端硬骨头**:任何高负载系统都要(并发、可靠性、状态、延迟成本、压测)。
> **桶二 · AI/Agent 特有硬骨头**:只有做 LLM 系统才会遇到(评测/LLMOps、上下文工程、Agent 控制论、RAG 质量、版本管理)。
>
> **个人 Agent 项目的差异化在桶二**——通用后端谁都能说会,但"我处理过 LLM 的不确定性 / 把评测工程化 / 做过上下文工程"是稀缺信号。

| # | 维度 | 桶 | 技术深度 | 个人项目加分 | 你的状态 |
|---|---|---|---|---|---|
| 1 | 高并发与扩展性 | 一 | 高 | ⭐⭐⭐ | ✅ 有审计 |
| 2 | 可靠性与韧性 | 一 | 高 | ⭐⭐⭐ | ⚠️ 缺熔断/降级 |
| 3 | 状态管理与一致性 | 一 | 高 | ⭐⭐ | ⚠️ 待 PostgresSaver |
| 4 | 延迟与成本工程 | 一 | 中高 | ⭐⭐⭐ 可量化 | ⚠️ 无缓存/分级 |
| 5 | 压测与容量规划 | 一 | 中 | ⭐⭐ 前置 | ❌ 无 query 基准 |
| 6 | 评估与 LLMOps | 二 | 高 | ⭐⭐⭐ 最稀缺 | ✅ 强,可补在线/CI |
| 7 | 上下文工程与记忆 | 二 | 中高 | ⭐⭐ | ⚠️ 部分 |
| 8 | Agent 控制论 | 二 | 中高 | ⭐⭐ | ✅ 有 cap/HITL |
| 9 | RAG 质量工程 | 二 | 中高 | ⭐⭐ | ✅ 强项 |
| 10 | 版本管理与可复现 | 二 | 中 | ⭐⭐ | ❌ prompt 无版本 |
| 11 | 渐进式发布 | 交付 | 中 | ⭐⭐ | ❌ |
| 12 | 优雅停机与零停机部署 | 交付 | 中高 | ⭐⭐ SSE 难点 | ❌ 无 lifespan |

**建议啃的顺序**(改动小、信号强优先):**2 可靠性 → 4 延迟成本 → 5 压测 → 6 评测工程化**。这四块叙事最完整、可量化、最能体现"工程"而非"调参"。

---

# 桶一 · 通用后端硬骨头

## 1. 高并发与扩展性（Concurrency & Scalability）

**是什么**:N 个请求同时来,谁真并行、谁串行、天花板在哪、怎么横向扩。

**真正的难点**:看穿"串行资源栈(serialization stack)"——event loop vs threadpool、GPU 单卡串行、连接池、单副本锁死。反直觉点:`async def` + 同步 GPU 调用**没有**阻塞事件循环(LangGraph 用 `run_in_executor` offload)。

**你的现状**:✅ 已有完整审计 [CONCURRENCY_REVIEW.md](CONCURRENCY_REVIEW.md)。瓶颈链:~20 线程池 → 单 GPU(1-at-a-time)→ LLM 账号配额 → 15 连接池 → 单进程。

**可啃子任务**(详见并发文档 §附录 B):
- [ ] P0 会话 `delete_thread` + TTL(防 OOM)
- [ ] P0 DB 写 offload / 异步引擎(消除事件循环阻塞)
- [ ] P1 `get_db` 改短生命周期(释放连接)
- [ ] P2 GPU 加锁 + Prometheus gauge

**面试钩子**:*"我能讲清为什么这个 async 端点里的同步 GPU 调用不阻塞事件循环——因为 LangGraph 把同步节点 offload 到线程池了。真正的并发墙是单 GPU 这个 serialization point,不是 event loop。"*

**术语**:event-loop blocking / threadpool saturation / serialization point / connection pool exhaustion / horizontal scaling

---

## 2. 可靠性与韧性（Reliability & Resilience）⭐

**是什么**:依赖会挂、网络会抖、LLM 会超时——系统怎么在故障下继续服务,而不是雪崩。

**真正的难点**:三层防御要协同设计,不是堆 try/except。
- **fail-fast(快速失败)**:熔断器(circuit breaker)——下游连续失败就直接拒,别让请求堆着等超时拖垮线程池
- **fail-soft(优雅降级)**:降级链(fallback chain)——主模型挂 → 备用模型 → 保守模板,**逐级兜底**
- **fail-safe(安全重试)**:幂等(idempotency)+ 退避抖动(backoff + jitter)+ 重试预算(retry budget)
- **超时预算传播(deadline budget / timeout propagation)**:一次问诊 8~15 个 LLM 调用要有**总闸**,把剩余预算传给每个下游,而不是每个单独 300s 各等各的
- **背压 + 准入控制(backpressure + admission control)**:满了快速返 429,而不是无声排队

**你的现状**:⚠️ 有 `with_retry` 重试,有 `agent_limits` 各种 cap;但**无熔断器、无模型降级路由、无单问诊全局 deadline**。并发文档还发现"SDK 嵌套重试 = 9 次往返"的重试风暴隐患。幂等是你的核心准则(✅)。

**可啃子任务**:
- [ ] `get_llm()` 设 `max_retries=0`,只保留一层重试(去掉 9 次往返)
- [ ] `with_retry` 加 jittered exponential backoff
- [ ] 给 DashScope/DeepSeek 调用包熔断器(pybreaker:连续失败 → 快速失败 → 半开试探)
- [ ] 实现降级链:主模型超时 → 备用模型 → §9.3 的 insufficient 模板(现在只有最后一级)
- [ ] 单问诊全局 deadline:进图时设总预算,每个节点检查剩余时间
- [ ] 准入控制:semaphore 限并发问诊数,超了返 429

**面试钩子**:*"我把可靠性拆成 fail-fast(熔断)、fail-soft(降级链)、fail-safe(幂等重试),并做 deadline propagation——单次问诊有总超时预算,不是每个 LLM 调用各等各的。"*

**术语**:circuit breaker / fallback chain / graceful degradation / deadline budget / timeout propagation / backoff + jitter / retry storm / backpressure / admission control / idempotency

---

## 3. 状态管理与一致性（State Management & Consistency）

**是什么**:Agent 是有状态的(stateful),会话跨多个 HTTP 请求、跨进程重启、跨副本——状态怎么存、怎么恢复、怎么不重复。

**真正的难点**:
- **检查点与恢复(checkpointing & recovery)**:`InMemorySaver → PostgresSaver`,重启不丢 in-flight 会话
- **交付语义(delivery semantics)**:SSE 断线重连(resume)时怎么保证 **exactly-once**——不重复写 `rag_trace`、不重复推进图。这是真问题,at-least-once 很容易,exactly-once 难
- **分布式状态(distributed state)**:多副本下用共享 checkpointer 让任意副本 resume 任意 `thread_id`(sticky session 的更优替代)

**你的现状**:⚠️ `InMemorySaver`(进程内存,重启全丢,锁死单副本);依赖 `PostgresSaver` 已在,差一张表 + Alembic 迁移。终态写 `rag_trace` 现在收口在 completed,降低脏数据但非严格幂等。

**可啃子任务**:
- [ ] `InMemorySaver` → `PostgresSaver`(建 `langgraph_checkpoints` 表 + Alembic)
- [ ] `rag_trace`/`conversation` 写入加幂等键(idempotency key),防 SSE 重连重复写
- [ ] 验证多副本下 resume 任意 session(横向扩展前置)

**面试钩子**:*"SSE 断线重连时我要保证 exactly-once 推进——LangGraph 的 checkpointer + 幂等写入键解决重复消费。"*

**术语**:checkpointing / state recovery / exactly-once vs at-least-once / idempotency key / distributed state / sticky session

---

## 4. 延迟与成本工程（Latency & Cost Engineering）⭐

**是什么**:云端 LLM 是真金白银 + 用户体验,怎么在 **质量/延迟/成本(quality / latency / cost)** 三角里做权衡。

**真正的难点**:这块**可量化**,是个人项目最佳展示位(能放 before/after 数字)。
- **尾延迟(p99 tail latency)**:看最慢 1%,不是平均。并行化"串行 12~30 个 sparse 查询"是典型
- **语义缓存(semantic caching)**:相似 query 命中缓存(对确定性调用如 enrichment/EL,不是主问诊流)
- **模型分级/级联(model-tier routing / cascading)**:抽槽/改写用便宜小模型,只有 ⑩ diagnose 用贵的 reasoning 模型——决策本身就是工程判断
- **Token 预算(token budgeting)** + **流式(streaming)** + **批处理(batching)**

**你的现状**:⚠️ 统计了 `token_usage`(✅,在 `state` + metrics),但**没有语义缓存、没有模型分级**,似乎一个模型走多数节点。Milvus 多路查询串行(并发文档已标)。

**可啃子任务**:
- [ ] 给确定性调用(enrichment/EL)加语义缓存,量化 token 节省
- [ ] 模型分级:轻节点(build_query/info_collect)换小模型,量化成本 vs top-1 变化
- [ ] 并行化 sparse 多路查询,量化 p99 降幅
- [ ] 接成本看板:按 session/user 归因 token 花费 + 预算护栏(超预算熔断)

**面试钩子**:*"我做 cost-latency-quality trade-off:用 model cascading 把单次问诊成本降 X%,top-1 不掉;并行化检索把 p99 从 A 降到 B。"*

**术语**:p99 tail latency / semantic caching / model-tier routing (cascading) / token budgeting / cost attribution / budget guardrails

---

## 5. 压测与容量规划（Load Testing & Capacity Planning）

**是什么**:不实测,所有"能扛多少并发"都是猜。这是上面所有量化结论的**前置**。

**真正的难点**:
- **基准测试(benchmarking)**:测出真实的 query-time 延迟(embedding encode、rerank、单次问诊端到端)。你仓库现在**只有 ingest-time 数字,没有 query-time 基准**(并发文档已标这是估算)
- **负载测试(load testing)**:k6/Locust 打 SSE 端点,找拐点(knee point)——延迟开始爆的并发数
- **容量规划(capacity planning)**:给定 GPU/配额,算出可服务的并发问诊数,反推扩容策略

**你的现状**:❌ 无 query-time 基准,无压测脚本。

**可啃子任务**:
- [ ] 写 query-time micro-benchmark:embedding encode / rerank 200 对 / 单次问诊端到端
- [ ] 用 k6 或 Locust 压 `/diagnose` SSE,画并发-延迟曲线找拐点
- [ ] 把估算的"5~20 并发天花板"换成实测数字,回填 [CONCURRENCY_REVIEW.md](CONCURRENCY_REVIEW.md)

**面试钩子**:*"我不拍脑袋说能扛多少——我用 k6 压 SSE 找到拐点,实测单 GPU 的问诊吞吐是每秒 X 个。"*

**术语**:benchmarking / load testing / knee point / capacity planning / SLI (Service Level Indicator)

---

# 桶二 · AI/Agent 特有硬骨头（你的差异化）

## 6. 评估与 LLMOps（Eval & LLMOps）⭐⭐ — 最值钱、最稀缺

**是什么**:LLM 输出非确定,"改了 prompt 是变好还是变坏"必须靠评测说话,不能靠感觉。

**真正的难点**:把"玄学调 prompt"变成"工程"。
- **离线评测(offline eval)**:golden dataset + RAGAS + LLM judge —— ✅ 你已经很强
- **回归门禁(eval-in-CI / regression gating)**:改 prompt/换模型 → 自动跑评测集,掉分就**卡住合并**。这一步把评测变成 CI 一环
- **在线评测(online eval)**:线上抽样实时打分,不只跑离线集
- **漂移监控(drift detection)**:模型/数据随时间变化,质量悄悄下滑的监控
- **护栏(guardrails)**:输出实时校验(格式/幻觉/groundedness/安全)
- **A/B + 影子流量(shadow traffic)**:新版本先吃影子流量对比再上

**你的现状**:✅ 离线评测体系扎实(top-1 93.5%、RAGAS、LLM judge、方法论文档、"不为分数调 prompt"的纪律——见 [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md))。缺在线/CI/drift。

**可啃子任务**:
- [ ] 把评测集接进 CI:PR 改 prompt → 自动跑 → 掉分阻断合并(regression gating)
- [ ] 加 guardrails:对 ⑩ diagnose 输出做 groundedness 校验(诊断是否被检索证据支撑)
- [ ] 线上抽样 + 离线 judge 打分(online eval 雏形)
- [ ] `diagnosis_feedback` 表接入:用真实反馈做 drift 信号

**面试钩子**:*"我把 eval 工程化了——评测集进 CI 做回归门禁,改 prompt 掉分直接卡合并;线上加 groundedness guardrail 防幻觉。大多数人会调 prompt,但很少有人把评测变成工程。"*

**术语**:offline/online eval / regression gating / eval-in-CI / drift detection / guardrails / groundedness / shadow traffic / golden dataset / LLMOps

---

## 7. 上下文工程与记忆（Context Engineering & Memory）

**是什么**:在有限 token 预算内,决定"塞什么进上下文"——以及 Agent 怎么记住跨会话的东西。

**真正的难点**:
- **上下文窗口管理(context window management)**:多轮追问累积时,旧轮次怎么裁/压。长会话不能无脑全塞
- **压缩/摘要(compaction / summarization)**:把长历史压成短摘要
- **记忆分层(memory tiers)**:工作记忆(working,单次会话状态)/ 情景记忆(episodic,这个病人历史)/ 语义记忆(semantic,知识库 + 沉淀经验)
- **记忆检索与淘汰(memory retrieval & eviction)**

**你的现状**:⚠️ 工作记忆做得细(`MedicalState` 12 维 HPI 槽,✅);有 [context/compressor.py](../src/rag/context/compressor.py)(方向对);**无情景记忆**(跨会话病人历史),多轮 followup 累积的上下文裁剪策略值得检查。

**可啃子任务**:
- [ ] 检查多轮 followup 是否无限累加进 prompt → 加截断/摘要策略
- [ ] 情景记忆雏形:同一 patient 跨会话调历史诊断(你已有 `patients`/`conversations` 表)
- [ ] 上下文 token 预算监控:每节点入口的 prompt token 埋点

**面试钩子**:*"Agent memory 分 working / episodic / semantic 三层;context engineering 的核心是在 token 预算内做 retrieval + compaction,而不是无脑全塞历史。"*

**术语**:context window management / compaction / summarization / working/episodic/semantic memory / memory retrieval / token budgeting / context engineering

---

## 8. Agent 控制论（Agent Control & Orchestration）

**是什么**:Agent 比普通 LLM 应用多一层"自主决策 + 循环 + 工具调用",最容易翻车,也最需要控制。

**真正的难点**:
- **循环控制 / 防失控(loop control / runaway prevention)**:Agent 无限循环烧 token 是真事故
- **确定性与可复现(determinism & reproducibility)**:同输入能否复现?LLM 本身非确定 → 固定 temperature/seed + 录制回放(record & replay)
- **工具可靠性(tool reliability)**:工具失败/超时/脏返回的处理,Agent 不能因一个工具挂了就崩
- **可调试 / 回放(traceability & replay)**:出错能否把那次 Agent 决策链完整重放定位
- **人在环(human-in-the-loop, HITL)**:关键决策插入人工

**你的现状**:✅ 有 `MAX_FOLLOWUP_ROUNDS`/`MAX_EXAM_ROUNDS` 硬上限(防失控,教科书级);✅ interrupt 驱动的追问/检查回传是 HITL 标准实现;✅ `rag_trace` 15 字段支持回放定位。缺:确定性测试(record & replay)、工具级错误处理体系化。

**可啃子任务**:
- [ ] record & replay:录一次真实问诊的 LLM 输入输出,离线回放调试图逻辑
- [ ] 工具/节点级统一错误处理策略(哪些可重试、哪些降级、哪些终止)
- [ ] 把 `rag_trace` 做成"决策链可视化"(一次问诊每节点的输入/输出/耗时)

**面试钩子**:*"Agent 最容易翻车的是失控循环和工具故障——我用 hard cap 防 runaway,用 interrupt 做 HITL,用 trace 做决策链回放。"*

**术语**:loop control / runaway prevention / determinism & reproducibility / record & replay / tool reliability / traceability / human-in-the-loop (HITL)

---

## 9. RAG 质量工程（RAG Quality Engineering）

**是什么**:检索质量直接决定生成质量——你的本行,也是强项。

**真正的难点 / 你的现状**:✅ 多路召回 + RRF 融合、reranker、parent-child chunking、实体链接(entity linking)、检索评测(Hit@K/NDCG@K,见 [RETRIEVAL_EVAL.md](RETRIEVAL_EVAL.md))都已做。

**可啃子任务**(企业级增量):
- [ ] 知识库新鲜度/同步监控(KB freshness):教材更新后如何增量重灌
- [ ] 索引维护(reindex / compaction)运维流程
- [ ] 检索失败兜底:召回为空/低分时的降级策略

**面试钩子**:*"我的 RAG 是多路召回 + RRF + rerank,检索侧主指标用 Hit@K/NDCG@K 而非 P@K——因为 P 是 K 的函数,现代 LLM 吃 20 chunks 不算噪音负担。"*

**术语**:multi-route retrieval / RRF (Reciprocal Rank Fusion) / reranking / parent-child chunking / entity linking / Hit@K / NDCG@K / KB freshness

---

## 10. 版本管理与可复现（Versioning & Reproducibility）

**是什么**:prompt、模型、数据都会变——出问题时能否定位"哪个版本的 prompt + 哪个模型 + 哪批数据"产生的这个结果。

**真正的难点**:
- **Prompt 版本管理(prompt versioning)**:prompt 是"代码",改动要可追溯、可回滚、可 A/B
- **模型版本固定(model pinning)**:`deepseek-chat` 这种别名背后会偷偷换模型 → 行为漂移。生产要 pin 具体版本
- **数据版本(data versioning)**:哪批知识库、哪个评测集
- **可复现(reproducibility)**:一条 `rag_trace` 能否还原出当时的完整配置

**你的现状**:❌ prompt 在 [src/prompts/](../src/prompts/) 无版本号;模型走 settings 别名未 pin;`rag_trace` 存了 `model_name` + `final_prompt`(✅ 部分可复现)。

**可啃子任务**:
- [ ] prompt 加版本号,`rag_trace` 落 prompt_version(改了能对应到结果)
- [ ] 模型 pin 到具体快照版本,别用滚动别名
- [ ] 评测报告关联 (prompt_version, model_version, dataset_version) 三元组

**面试钩子**:*"prompt 是代码,要版本化 + 回滚;模型别名会偷偷漂移,生产必须 pin。我的 trace 落了 prompt/model 版本,任何一次诊断都能复现当时的配置。"*

**术语**:prompt versioning / model pinning / data versioning / reproducibility / config snapshot

---

# 交付与发布（中等优先，但有真难点）

## 11. 渐进式发布（Progressive Delivery）

**是什么**:新版本不要一把全量上,先小流量验证。

**真正的难点**:
- **特征开关(feature flags)**:新逻辑藏在开关后,可随时关(kill switch)
- **金丝雀/灰度(canary release)**:5% 流量先上,看指标再放量
- **影子流量(shadow traffic)**:新版本吃复制流量但不返给用户,对比质量
- **A/B 测试**:两版本同时跑,数据决定胜出

**你的现状**:❌ 无 feature flag,无灰度机制。

**可啃子任务**:
- [ ] 引入 feature flag(哪怕 settings 开关起步):新 prompt/模型藏开关后
- [ ] kill switch:出问题一键回退到保守路径

**面试钩子**:*"改诊断逻辑我不直接全量——feature flag + canary,出事一键 kill switch。"*

**术语**:feature flags / canary release / shadow traffic / A/B testing / kill switch

---

## 12. 优雅停机与零停机部署（Graceful Shutdown & Zero-Downtime Deploy）

**是什么**:部署/重启时,正在跑的请求别被硬切断。

**真正的难点**(你这场景特别有料):
- **优雅停机(graceful shutdown)**:收到 SIGTERM 后,**排空(drain)**正在跑的请求再退出
- **难点放大**:你的 `/diagnose` 是**长连接 SSE(最长 600s)** + **interrupt 中断会话**——硬重启会断流 + 丢中断状态(`InMemorySaver`)。优雅停机要么等流跑完,要么把状态落 `PostgresSaver` 后再退
- **零停机部署(zero-downtime / rolling deploy)**:配合健康检查 + 排空 + 新副本就绪再切流量

**你的现状**:❌ 无 `lifespan`/`shutdown` 钩子(扫描确认);✅ 已有 `/healthz`+`/readyz`(K8s 探针就绪,见 [health.py](../src/api/routes/health.py))。

**可啃子任务**:
- [ ] FastAPI `lifespan` 加 shutdown:SIGTERM → 停止接新请求 → drain in-flight SSE → 退出
- [ ] 配合 `PostgresSaver`:重启前 in-flight 中断会话已落库,重启后可恢复
- [ ] readiness 探针在 drain 期间返 not-ready,让 LB 停止打新流量

**面试钩子**:*"我这是长连接 SSE + 有状态中断会话,优雅停机不能简单等连接关——要么 drain 600s 的流,要么把 checkpoint 落库再退,否则重启丢中断态。"*

**术语**:graceful shutdown / connection draining / SIGTERM handling / zero-downtime deploy / rolling update / readiness probe

---

# 刻意跳过（个人项目低性价比）

这些**企业必须做、但技术含金量低**,是流程/体力活,个人项目做了难加分——知道它们存在、面试能说出"为什么这是合规问题不是技术问题"即可:

- **合规与数据治理(compliance & governance)**:医疗 PHI、PDPA(新加坡)/HIPAA、数据保留与删除权。*真做要上加密/脱敏/审计,但本质是流程驱动。*
- **密钥托管(secrets management)**:Vault / Secrets Manager + 轮转。*技术上就是"别把密钥写进代码",不难。*
- **基础权限(AuthN/AuthZ plumbing)**:RBAC、细粒度权限。*你已有 JWT,够 demo。*
- **多租户(multi-tenancy)**:除非要做 SaaS,否则个人项目不必。*若做,难点在租户隔离 + 按租户限流/计费,那时它才变成技术活。*

> 面试表达:*"合规和密钥这类我归为'tedious not hard'——它们是流程驱动的工程,我会做但不是我想用来证明技术深度的地方。我的技术亮点放在可靠性、评测工程化和延迟成本优化上。"*

---

## 附录：术语总表（中英对照）

| 中文 | English |
|---|---|
| 串行化资源 | serialization point |
| 熔断器 | circuit breaker |
| 降级链 / 优雅降级 | fallback chain / graceful degradation |
| 超时预算传播 | deadline budget / timeout propagation |
| 退避 + 抖动 | backoff + jitter |
| 重试风暴 | retry storm |
| 背压 / 准入控制 | backpressure / admission control |
| 幂等 / 幂等键 | idempotency / idempotency key |
| 检查点与恢复 | checkpointing & recovery |
| 精确一次 / 至少一次 | exactly-once / at-least-once |
| 尾延迟 | p99 tail latency |
| 语义缓存 | semantic caching |
| 模型分级 / 级联 | model-tier routing / cascading |
| 成本归因 / 预算护栏 | cost attribution / budget guardrails |
| 压测 / 拐点 | load testing / knee point |
| 容量规划 | capacity planning |
| 回归门禁 | regression gating / eval-in-CI |
| 在线评测 / 漂移监控 | online eval / drift detection |
| 护栏 / 接地性 | guardrails / groundedness |
| 影子流量 | shadow traffic |
| 上下文工程 | context engineering |
| 压缩 / 摘要 | compaction / summarization |
| 工作 / 情景 / 语义记忆 | working / episodic / semantic memory |
| 循环控制 / 防失控 | loop control / runaway prevention |
| 确定性与可复现 | determinism & reproducibility |
| 录制回放 | record & replay |
| 人在环 | human-in-the-loop (HITL) |
| Prompt 版本管理 | prompt versioning |
| 模型固定 | model pinning |
| 特征开关 / 金丝雀 | feature flags / canary release |
| 优雅停机 / 排空 | graceful shutdown / connection draining |
| 零停机部署 | zero-downtime deploy / rolling update |
| SLO / 错误预算 | SLO / error budget |
| 单点故障 | SPOF (single point of failure) |
