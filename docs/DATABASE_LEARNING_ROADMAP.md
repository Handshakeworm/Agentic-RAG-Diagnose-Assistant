# 数据库必学路线图 — 面向 AI 应用开发求职

> **用途**:为求职 **AI 应用开发(AI Application Developer)** 梳理的数据库必学内容。最大特点:**用本项目当教材**——本仓库已实战 PostgreSQL + Milvus(向量库)+ Redis 三套数据库,每个知识点都标了"你代码里的抓手"(均已对照真实代码核实),对着学最快。
>
> **术语约定**:中文为主,关键概念附英文行业术语,方便英文语境(面试/文档)表达。
>
> 编写日期:2026-06-02 · 已对照 `main` 代码核实抓手

---

## 0. 先划范围:AI 应用开发的数据库 ≠ 数据工程

别学偏。求 AI 应用开发,**不需要**深啃 Spark / 数仓 ETL / Flink(那是**数据工程师 Data Engineer** 的方向)。你要掌握的是:

- **在线业务数据库(operational / OLTP databases)**:PostgreSQL 这类关系型
- **向量检索(vector retrieval)**:RAG / Agent 的核心差异化
- **缓存(caching / KV store)**:Redis

> **OLTP vs OLAP**(面试可能问):OLTP = 在线事务处理(高频小事务,你的业务库);OLAP = 在线分析处理(大批量分析查询,数仓)。AI 应用开发主战场是 OLTP + 向量 + 缓存。

---

## 优先级总览

| 层 | 内容 | 学到什么程度 | 你项目里用到 |
|---|---|---|---|
| 🥇 地基 | SQL + PostgreSQL | **讲透**(必考) | PG:JSONB / 连接池 / 幂等 UPSERT |
| 🥈 AI 核心 | 向量数据库 + 检索 | **讲透**(差异化) | Milvus:HNSW / BM25 / RRF |
| 🥉 支撑 | Redis / 缓存 | **能聊**(高频) | ZSET 限流 / cache-aside |
| ⭐ 锦上添花 | 图库 / 时序库 / 数仓 | 先不碰 | — |

> **下面「必学点」表的优先级标记**(按 AI 应用开发实习岗):🔴 必学(高频 + 面试必问)· 🟡 了解即可(对本岗低 ROI,给一句"是什么"就够)· 🟢 顺带就会(简单,别专门花时间)。

> **为什么没有「Agent 状态 / 记忆持久化」**:它是「用数据库解决的应用问题」,不是一类数据库或 DB 概念——概念属框架(LangGraph)+ Agent 设计。放 [ENTERPRISE_READINESS.md](ENTERPRISE_READINESS.md) §3(状态一致性)/ §7(记忆),不在本 DB 路线图。

---

# 🥇 地基层:SQL + PostgreSQL

任何后端/AI 岗的入场券,**最该先啃、最常被问**。

> 📒 **逐点笔记**(配简要例子)在 [sql-postgres/](sql-postgres/) 文件夹:
> - [① 增删查改(CRUD)+ UPSERT](sql-postgres/01-crud.md)(含 查 SELECT 的 WHERE 条件家族)✅

## 必学点

| 优先级 | 主题 | English | 说明 |
|---|---|---|---|
| 🔴 | 增删查改(基础查询) | CRUD | `SELECT`/`INSERT`/`UPDATE`/`DELETE`,地基。已学 ✅ [sql-postgres/01-crud.md](sql-postgres/01-crud.md) |
| 🔴 | 多表关联 | JOIN | inner / left / right / full,日常高频 |
| 🔴 | 聚合 + 分组 | GROUP BY / HAVING | `COUNT`/`SUM`/`AVG`;`WHERE`(组前)vs `HAVING`(组后) |
| 🔴 | 子查询 | subquery | 嵌套查询(标量 / `IN` / `EXISTS`) |
| 🟡 | 窗口函数 | window functions | **是什么**:不合并行地按"窗口"给每行算排名/累计(`OVER (PARTITION BY ...)`)。偏数据分析岗,AI 应用少用,**了解即可** |
| 🟢 | 公共表表达式 | CTE | **是什么**:`WITH 名 AS (子查询)` 给子查询起名、让复杂查询更好读(还能递归)。简单,**顺带会** |
| 🔴 | 索引 | indexing | B-tree、复合索引、覆盖索引、最左前缀、何时失效 |
| 🟢 | 查询计划 | EXPLAIN | **是什么**:SQL 前加 `EXPLAIN` 看数据库打算怎么执行(走索引 vs 全表扫)。**配合索引会读即可** |
| 🔴 | 事务 + ACID | transactions | 原子性/一致性/隔离性/持久性 |
| 🔴 | 隔离级别 | isolation levels | 知道默认 `READ COMMITTED` + 三种异常(脏读/不可重复读/幻读)即可;**serializable 内核别深抠** |
| 🟡 | MVCC | MVCC | **一句话**:PG 每行存多版本 → 读不阻塞写、写不阻塞读。**记这句即可**,内部实现偏 DBA |
| 🔴 | 锁 + 死锁 | locking | 重点:乐观锁 vs 悲观锁(概念 + 适用场景) |
| 🔴 | 连接池 | connection pooling | pool_size / overflow / 耗尽后阻塞超时 |
| 🔴 | 范式 vs 反范式 | normalization | schema 设计取舍 |
| 🔴 | JSONB | JSONB | 半结构化数据原样存(你 `raw_documents` 在用) |
| 🟡 | PG 全文检索 | full-text search | **是什么**:PG 内置关键词搜索(`tsvector`/`@@`)。AI 应用交给向量库 BM25,**可跳过** |
| 🔴 | N+1 查询 | N+1 query problem | ORM 常见坑 |
| 🔴 | 幂等写入 | idempotent write | `INSERT ... ON CONFLICT`(UPSERT),已学 ✅ |

## 你项目里的抓手(已核实)

- **JSONB**:[../src/db/postgres/models.py](../src/db/postgres/models.py) 的 `raw_documents` 表,3 个 JSONB 字段(`content_list` / `middle_data` / `model_data`)原样存 MinerU 解析产物
- **幂等 UPSERT**:同文件 `upsert_raw_document()` 用 `INSERT ... ON CONFLICT (source_id) DO UPDATE`——这是幂等写入的教科书写法,可直接讲
- **连接池**:[../src/db/postgres/connection.py](../src/db/postgres/connection.py) `QueuePool(pool_size=5, max_overflow=10)` + `pool_pre_ping`;耗尽场景见 [CONCURRENCY_REVIEW.md](CONCURRENCY_REVIEW.md) §3/§4(你已做过审计,能讲得比多数候选人深)
- **事务边界**:`session_scope()` 上下文管理器(正常 commit / 异常 rollback / 必 close)

## ⚠️ spec/code gap(诚实记录)

CLAUDE.md / DEV_SPEC 描述 `raw_documents` 为 "JSONB + **GIN**",但**代码里没有任何 GIN 索引实现**(ORM 模型、init_db、alembic 全无),连自定义二级索引都没建。所以:
- 别在简历/面试说"我建了 GIN 索引"——目前不属实
- 这反而是个**真实可做的练手项**:给 JSONB 字段加 GIN 索引(`CREATE INDEX ... USING GIN (content_list)`),用 `EXPLAIN ANALYZE` 对比加索引前后查询计划。学索引 + 补上 gap 一举两得。

## 面试高频题

- 索引为什么能加速?B-tree 适合什么、不适合什么?什么情况下索引失效?
- 事务隔离级别有哪几种?各解决什么问题(脏读/不可重复读/幻读)?
- 乐观锁 vs 悲观锁的区别和适用场景?
- 连接池满了会发生什么?(→ 你能用本项目实例答)
- JSONB 和 JSON 的区别?什么时候该用 JSONB 而不是拆成列?

---

# 🥈 AI 核心层:向量数据库 + 检索

AI 应用区别于普通后端的地方,**面试官最想听**。

## 必学点

| 优先级 | 主题 | English | 说明 |
|---|---|---|---|
| 🔴 | 向量嵌入存储 | embeddings storage | 一条记录 = 向量 + 元数据 |
| 🔴 | 相似度度量 | similarity metric | cosine / dot product / L2;归一化后 cosine≈dot |
| 🔴 | ANN 近似最近邻 | approximate nearest neighbor | 为什么不做暴力精确搜索:维度高、数据大 |
| 🔴 | 向量索引类型 | vector index | **HNSW**(你在用,图索引)/ IVF(倒排聚类)/ FLAT(暴力);DiskANN 较冷门,知道名字即可 |
| 🔴 | 召回 vs 延迟取舍 | recall vs latency trade-off | ANN 核心:用准确率换速度 |
| 🟡 | 量化压缩 | quantization | **是什么**:压缩向量存储,用一点精度换内存/速度。**了解"精度换资源"概念即可** |
| 🔴 | 元数据过滤 | metadata filtering | 向量搜 + 标量条件(pre-filter vs post-filter) |
| 🔴 | 混合检索 | hybrid search | dense(语义)+ sparse(BM25 关键词) |
| 🔴 | RRF 融合 | reciprocal rank fusion | 多路检索结果按排名融合 |
| 🔴 | 选型对比 | — | pgvector vs Milvus vs Pinecone / Qdrant / Weaviate / Chroma |

## 你项目里的抓手(已核实)

- **HNSW 索引**:[../config/milvus_schema.py](../config/milvus_schema.py) dense 向量用 `index_type="HNSW"` + `metric_type="COSINE"`(`docs_collection` 和 `terms_collection` 都是)。**注意:你项目用的是 HNSW,不是 IVF**——IVF 是备选,面试可对比但别说错
- **HNSW 检索约束**:[../src/db/milvus/docs_collection.py](../src/db/milvus/docs_collection.py) 注释明确 `ef ≥ k`,否则 Milvus 报错——这种实操细节面试很加分
- **混合检索**:dense(HNSW)+ sparse(`SPARSE_INVERTED_INDEX` + Milvus 内置 BM25 Function)双路,见 `docs_collection.py` 的 `search_dense` / `search_sparse_bm25`
- **RRF 融合**:[../src/rag/retrieval/fusion.py](../src/rag/retrieval/fusion.py) 把多路结果按排名倒数融合 + 按 `source_chunk_id` 去重
- **标量索引**:`source_chunk_id` / `vector_type` / `source_id` 用 `INVERTED` 索引(支持元数据过滤,如只搜 `vector_type="original"`)
- **维度**:4096(Qwen3-Embedding-8B);`terms_collection` 同维做实体链接(entity linking)
- **多向量策略**:同一 chunk 存 original + summary + question 多个向量(`vector_type` 区分)

## 面试高频题

- HNSW 的原理?为什么比暴力搜索快?它的内存代价是什么?
- HNSW vs IVF 怎么选?(→ 你用 HNSW,能讲为什么)
- 向量库怎么做带过滤的检索?pre-filter 和 post-filter 的区别?
- dense 检索和 sparse(BM25)检索各擅长什么?为什么要混合?怎么融合(RRF)?
- **什么时候用 pgvector 就够、什么时候上 Milvus/专用向量库?**(经典题:数据量、QPS、是否需要独立扩展、运维成本)
- cosine / dot product / L2 怎么选?

---

# 🥉 支撑层:Redis / 缓存

高频考点,而且你有真实代码可讲。

## 必学点

| 主题 | English | 备注 |
|---|---|---|
| 数据结构 | data structures | String / Hash / List / Set / **ZSet(sorted set)** / Stream / Bitmap |
| **缓存模式** | caching patterns | **cache-aside(旁路,最常用)** / read-through / write-through / write-behind |
| TTL + 淘汰策略 | TTL / eviction | LRU / LFU / volatile-ttl |
| **缓存三大问题** | — | **穿透 penetration** / **击穿 breakdown(hotkey)** / **雪崩 avalanche** + 一致性 consistency |
| 固定窗口 vs 滑动窗口 | fixed vs sliding window | 限流算法 |
| 应用场景 | — | 分布式锁(distributed lock)/ 限流(rate limiting)/ 会话 / 消息队列(pub-sub / Stream)|

## 你项目里的抓手(已核实)

- **ZSET 滑动窗口限流**:[../src/db/redis/rate_limit_backend.py](../src/db/redis/rate_limit_backend.py) 用 `ZREMRANGEBYSCORE`(清窗外)+ `ZCARD`(计数)+ `ZADD`,~5 命令/请求。**代码注释明确写了为什么不用 `INCR+EXPIRE` 固定窗口:整点会有 2 倍配额突刺**——这是滑动 vs 固定窗口的绝佳面试素材
- **fail-open**:Redis 挂了限流放行(优雅降级),见同文件 docstring
- **cache-aside + TTL**:[../src/db/redis/cache.py](../src/db/redis/cache.py) `get_config_cached()`——读缓存→未命中→`loader()`→`SETEX` 写回(60s TTL)。标准旁路缓存
- **缓存穿透的反向取舍**:同文件 `loader()` 返回 None 时**故意不写缓存**——这是"缓存空值(穿透防护)" 与 "配置新鲜度" 的权衡:不缓存 None 是为了管理员补配置后不用等 60s 生效。能讲清这个取舍 = 真懂缓存

## 面试高频题

- 缓存穿透 / 击穿 / 雪崩分别是什么?怎么解?
  - 穿透:查不存在的 key,每次都打 DB → 缓存空值(短 TTL)或布隆过滤器(bloom filter)
  - 击穿:热点 key 过期瞬间大量请求打 DB → 互斥锁重建 / 热点 key 不过期
  - 雪崩:大量 key 同时过期 → TTL 加随机抖动
- 怎么用 Redis 实现限流?固定窗口和滑动窗口的区别?(→ 你能用本项目答)
- 缓存和数据库的一致性怎么保证?(先删缓存还是先更 DB?延迟双删?)
- ZSet 能做什么?(排行榜、滑动窗口、延时队列)

---

# ⭐ 锦上添花(先不碰)

知道存在、能说出适用场景即可,**别花主力时间**:

- **图数据库(graph DB,Neo4j)**:GraphRAG、知识图谱
- **时序数据库(time-series,TimescaleDB / InfluxDB)**:监控指标、IoT
- **文档数据库(document DB,MongoDB)**:灵活 schema
- **数据仓库/湖(data warehouse / lake,BigQuery / Snowflake)**:偏数据工程

---

# 学习策略:把项目当教材(最高效)

别从零啃《数据库系统概念》。顺序:

1. **先把项目里每个表/索引/查询「为什么这么设计」讲清楚**——读 [../src/db/](../src/db/) 下 PG/Milvus/Redis 代码,每个设计回答"换种方式行不行、代价是什么"
2. **讲不清的地方反查理论**——讲到连接池就补事务/锁;讲到 HNSW 就补 ANN 原理;讲到限流就补滑动 vs 固定窗口
3. **每块配一道面试题自测**(见各层"面试高频题")
4. **动手验证**:
   - PG:给 JSONB 加 GIN 索引,`EXPLAIN ANALYZE` 对比前后(顺便补上那个 spec/code gap)
   - Redis:`redis-cli` 手玩 ZSET 滑动窗口
   - Milvus:换索引参数(ef / HNSW vs FLAT)看召回 vs 延迟变化

学完你不是"背过数据库",而是"做过一个三数据库的 AI 系统并能讲透取舍"——这正是 AI 应用岗想要的。

---

# 术语小抄(中英对照)

| 中文 | English |
|---|---|
| 在线事务处理 / 在线分析处理 | OLTP / OLAP |
| 窗口函数 | window functions |
| 复合 / 覆盖索引 | composite / covering index |
| 查询计划 / 执行计划 | query plan / execution plan |
| 事务 / 隔离级别 | transactions / isolation levels |
| 脏读 / 不可重复读 / 幻读 | dirty read / non-repeatable read / phantom read |
| 多版本并发控制 | MVCC |
| 乐观锁 / 悲观锁 | optimistic / pessimistic locking |
| 死锁 | deadlock |
| 连接池 | connection pooling |
| 范式 / 反范式 | normalization / denormalization |
| 幂等写入 / 插入更新 | idempotent write / UPSERT |
| N+1 查询问题 | N+1 query problem |
| 近似最近邻 | ANN (Approximate Nearest Neighbor) |
| 余弦 / 内积 / 欧氏距离 | cosine / dot product / L2 (Euclidean) |
| 召回率 vs 延迟 | recall vs latency |
| 量化压缩 | quantization (PQ / SQ) |
| 元数据过滤 | metadata filtering |
| 混合检索 | hybrid search |
| 倒数排名融合 | RRF (Reciprocal Rank Fusion) |
| 缓存旁路 | cache-aside |
| 缓存穿透 / 击穿 / 雪崩 | cache penetration / breakdown / avalanche |
| 固定 / 滑动窗口 | fixed / sliding window |
| 分布式锁 | distributed lock |
| 优雅降级 / 故障放行 | graceful degradation / fail-open |
