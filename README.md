[简体中文](README.md) | [English](README.en.md)

# Agentic-RAG Diagnose Assistant

> 患者侧症状自查与初诊判断系统 — 基于 LangGraph Agent + 多路检索 RAG。
>
> **个人练手项目**(非生产部署),涵盖数据工程 → ML 推理 → Agent 编排 → 后端 → 基础设施 → 评估的全栈端到端实现。
> 设计与实现完全 spec-driven,单一事实源:[DEV_SPEC.md](DEV_SPEC.md)(4976 行)。

---

## 演示

<div align="center">
  <a href="https://streamable.com/kjmixi" title="点击观看 1 分钟 Demo">
    <img src="assets/demo-cover.png" alt="AI 医生 Demo — 点击观看完整视频" width="720"/>
  </a>
  <br/>
  <sub>▶ 点击图片跳转 Streamable 观看 1 分钟端到端对话(主诉 → 多轮追问 → 报告上传 → 诊断 → 用药建议)</sub>
</div>

---

## 评测成绩(62 case 执业医考题,2026-05-17)

> 两层评测:**① RAG 检索**(召回的医学文献是否覆盖正确章节)→ **② 诊断**(LLM 基于召回 chunk 给出诊断的准确率)

### ① RAG 检索质量(LLM Judge 对 parent 评 0~3 分作 ground truth)

| 指标 | 成绩 | 含义 |
|---|---|---|
| **Hit@20(≥3 高相关)** | **100%** | 62/62 case **0 漏掉高相关章节** |
| **NDCG@20** | **0.774** | 排序质量(医疗专域 0.6~0.8 区间上中段) |
| **MRR(≥2)** | **0.954** | 几乎所有 case Top-1 即命中中度相关 |
| Spearman ρ vs LLM | +0.708 | RAG 排序与 LLM 判断高度一致 |

### ② 诊断准确率(LLM 基于 RAG Top-20 给出诊断,Judge 评 gold ↔ LLM 等价性)

> **多 gold case**:同一患者并存多个疾病/合并症的 case(如 "脾破裂 + 肋骨骨折"、"冠心病急性心梗 + 急性左心衰竭" — 62 case 中约半数属此类)

| 指标 | 成绩 | 含义 |
|---|---|---|
| **Top-1 临床命中率** | **93.5%** | LLM 第一选项就是 gold 临床等价(58/62)|
| **Top-2 命中率** | **100%** | gold 主诊断必在前 2 — **62/62 几乎不漏主诊断方向** |
| **0 主诊断方向错** | **0/62** | 所有 case 都给出正确诊断方向(无 `none`)|
| **多 gold 平均覆盖率** | **86.7%** | 多 gold case 平均 86.7% 的 gold 被 LLM 列出 |
| 多 gold 全召回率 | 72.6% | 多 gold case 所有 gold 都被覆盖(45/62)|

→ 详细方法学 / match_type 分布 / 关键设计决策见 [评测结果](#评测结果) · 全检索层细节见 [RETRIEVAL_EVAL.md](docs/RETRIEVAL_EVAL.md)

---

## 目录

- [项目定位](#项目定位)
- [设计亮点](#设计亮点)
- [系统架构](#系统架构)
- [Agent 工作流](#agent-工作流)
- [RAG 流水线](#rag-流水线)
- [数据层](#数据层)
- [技术栈与选型理由](#技术栈与选型理由)
- [监控与可观测性](#监控与可观测性)
- [快速开始](#快速开始)
- [评测结果](#评测结果)
- [项目进度](#项目进度)
- [目录结构](#目录结构)
- [文档导航](#文档导航)
- [开发约定](#开发约定)

---

## 项目定位

**做什么**:用户用自然语言描述症状,系统通过多轮追问澄清病史、检索 13 本医学教材知识库、**给出初步诊断 + 鉴别诊断方向 + 进一步检查建议**。62 case 执业医考题评测:**Top-1 临床等价命中率 93.5%、Top-2 100%、0 主诊断方向错**(详见下方"评测结果")。

**给谁用**:为患者提供初诊判断与就医引导。所有 LLM 输出经 `safety_gate` 节点二次过滤,**不开具处方剂量、不替代专业医生面诊** — 系统的角色是"给出医生可能想到的诊断方向 + 建议如何进一步检查",最终诊断与治疗仍以执业医师为准。

**不做什么**:影像直接解读、外科手术规划、儿科特化、药物剂量计算 — 这些场景需要专业模型或医生现场判断,超出本项目范围。

---

## 设计亮点

### 1. 全栈端到端自实现

| 层 | 我做了什么 |
|---|---|
| 数据处理 / RAG | • MinerU 解析医学教材(13 本计划,目前已完成 12 本,共 264948 文档块),每本教科书专配一套脚本精细化清洗切分,12 本均零边界丢失<br>• 父子两级分块:外层按章节切出父块、内层按 token 切出子块<br>• 每块产出原文 + LLM enrichment 摘要 + LLM enrichment 3个假设患者问题, + BM25 倒排,共 26054 块 / 129810 向量入 Milvus<br>• 先 PG 后 Milvus 双写,幂等可重跑 + 自动清理孤儿块 |
| Embedding / Reranker | • Qwen3-Embedding-8B(8.5GB)+ BGE-Reranker-v2-minicpm-layerwise(2.6GB) INT8 量化单卡 16GB 共显<br>• 精排 layerwise 早退加速;失败 / 超时回退到召回原序 |
| Agent | • LangGraph **17 节点 + 4 条件分支**组织成状态机<br>• **12 维 HPI 结构化主动问诊**:⑤ 节点 1 LLM 调用同时出追问(slot 维度填补 + open 开放式兜底)+ unaskable 粗筛,信息已足时直接跳诊断,而非被动等用户描述<br>• Human in loop: 暂停等待用户输入(多轮澄清病史 / 上传补充检查报告)<br>• 诊断 1 步 LLM 出 ranking + retained_unaskable(对齐 RAG 评测口径 top1 93.5%),失败重试 3 次仍报错则早停告知"信息不足以确诊"<br>• 最终输出经独立安全过滤,规避处方剂量与确诊口吻<br>• **LLM 三档路由**:7 处 flash(翻译/抽取)+ 3 处 pro(鉴别推理/安全/用药)+ 2 处 vision(报告解析 / ⑩ 多模态诊断),详见下方 §6 |
| 后端 | • FastAPI + JWT 实现注册 / 登录 / 角色守卫<br>• 限流先抽象后实现:单机内存版可换多副本 Redis 共享版,业务代码不动<br>• PostgreSQL 20 表 + Alembic 6 次迁移<br>• 每次问诊同事务写响应 + 15 字段审计链路(90 天保留) |
| 基础设施 | • Docker Compose 13 容器一键启动<br>• Prometheus 6 监控目标 + 11 类业务指标(LLM 健康度 / 上下文长度 / PG·Redis·Milvus 三层依赖)<br>• Grafana 启动自动加载 2 仪表盘(应用 + 硬件);日志面板点击跳数据库审计详情<br>• 每请求一个 trace ID 串日志 / 审计 / 监控三路<br>• **基础设施降级哲学**:Redis 挂回源 PG / 限流 fail-open / Reranker 超时回退原序,故障半径控死在一层 |
| 工程过程 | • Spec-driven 协作开发:[DEV_SPEC.md](DEV_SPEC.md) 唯一事实源,[CLAUDE.md](CLAUDE.md) 锚定开发红线<br>• 我做架构与取舍判断,Claude 落地代码 + 反向同步 §8.4 进度<br>• 测试 333 单元 + 95 集成 PASS |

### 2. 多路检索 + 多向量索引(单阶段加权 RRF)

不是常见的"先 dense 再 sparse 后 rerank"流水,而是召回阶段同时玩两个维度的"多":

**多路加权 RRF**(state 多字段 sparse + 1 路 dense,RETRIEVAL_EVAL §2/§4 评测确定):
- Sparse 路 **多字段直采**:
  - 来源 A — state 结构化字段:`chief_complaint` + `present_illness_slots` 6 单值字段(trigger/location/nature/severity/duration_pattern/onset_mode)+ 3 list 字段(associated_symptoms/aggravating/relieving)
  - 来源 B — 报告语义信号:`report_findings.positive_findings` 全加 + `impressions` 阴性过滤(`(-)`/正常/阴性/未见/无异常)
  - 实测 62 case 平均 21.8 条 sparse 词袋
- 每条词袋 = 1 次独立 BM25(不是拼成 1 个 query),N 条 = N 次查询
- Dense 路 1 次 ANN
- (N+1) 路一起做**加权 RRF 融合**:`dense_weight = max(1, N_sparse/5)` 平衡 sparse 多路对 dense 的挤兑(RETRIEVAL_EVAL §4);sparse 多路命中累加保留"跨模态自调节"特性

**多向量索引**(每个 chunk 入库 5 路向量):

| vector_type | 主要捕获 | 弥合 |
|---|---|---|
| `original` | 整体语义 + BM25 关键词 | 患者描述与原文术语 |
| `summary` | 方法论 / 综述类查询 | "X 是什么 / 用来干嘛" |
| `question` × 3 | 患者口语 + 场景化代入 | "我爸爸喘得厉害爬楼困难" |

**单条向量的任务不是回答 query,只是让 chunk 进入 Top-200 候选池**;Reranker 看完整 `chunk_raw_text` 重排,与命中哪条向量无关。详见 [DEV_SPEC §3.2.2 / §3.1.5.1](DEV_SPEC.md#322-召回)。

### 3. Small-to-Big 父子索引

医疗知识天然分层(疾病 → 亚型 → 治疗方案 → 剂量/禁忌)。小块精确命中"剂量"时禁忌可能在同节另一小块,直接送 LLM 会**危险遗漏**。本项目:

- 父块按"书的目录权威清单 + 节内三遍切【】+(一)+1."切,**不向量化**(`embedding_status='skip'`),仅作 Small-to-Big 展开 context
- 子块按 size 驱动累积切(目标 ~600 字),独立向量化
- 召回命中子块 → 按 `parent_chunk_id` 回查父块全文一并送 LLM
- **不变量**:`total_parent_len == total_child_len`(POC 验证 12 本 mismatch=0)

详见 [DEV_SPEC §3.1.2](DEV_SPEC.md#312-chunking目录权威清单--三遍切--size-驱动子块)。

### 4. 单卡 16GB 共显 Embedding + Reranker

RTX 5070 Ti 16GB 同时承载 Embedding 8B(~8.5GB)+ Reranker 2.4B(~2.6GB),靠 **bitsandbytes INT8 量化**把两个模型压到 ~11GB,留 4.6GB 给推理激活值。LLM 推理走云端 API(DeepSeek 主链 + DashScope 多模态)不占 GPU,所以 16GB 全留给本地两个 Encoder 模型。详见 [DEV_SPEC §2.1 / §2.3](DEV_SPEC.md#21-embedding-模型选型)。

### 5. 多模态摄取流水线(文本 + 表格 + 图表)

不是只切文本 — `chunks` 表用 **单行多列 + chunk_type** 设计统一容纳三类:

- `child` / `parent`:正文文本(`chunk_raw_text` 主)
- `table`:caption + html + footnote(BM25)+ text LLM 看 html 生成 100-300 字 `medical_statement`(dense `original`)
- `figure`:caption + footnote(BM25)+ vision LLM 看截图生成 `medical_statement`(dense `original`)

图表 chunk 共享父块的 `heading_path_id`,在召回时通过 **同节图表跟随父块**(规则 3,封顶 5 条)的回拉机制保证图表必然被带出来。详见 [DEV_SPEC §3.1.2 / §3.2.3](DEV_SPEC.md#312-chunking目录权威清单--三遍切--size-驱动子块)。

### 6. LLM 按能力路由(三档分流:flash / pro / vision)

不是一个 LLM 干所有事 — **按"任务复杂度 + 是否要看图"三档分流**,12 处 LLM 调用分到 3 个模型上:

| 档位 | 模型 | 适用任务 | 节点(共 12 处) |
|---|---|---|---|
| **flash**(轻型, ~1-5s) | DeepSeek-v4-flash | 翻译 / 格式整合 / 一次性抽取(语义浅) | ① info_collect, ② build_query(dense 改写), ⑤ generate_followup(Step A holistic + Step B 拼句), ⑦ process_followup, ⑧a recommend_exam, ⑬ format_response(7 处) |
| **pro**(重型, ~5-20s) | DeepSeek-v4-pro(thinking) | 鉴别诊断推理 / 安全推理 / 治疗方案 | ④ select_discriminative_symptom, ⑪ safety_gate, ⑫ generate_advice(3 处) |
| **vision**(多模态, ~30-120s) | DashScope qwen3.5-plus | 看图(报告 PDF / 影像截图)+ 临床推理 | ①.5 analyze_initial_reports, ⑨ process_exam_result(共享 parse_reports), ⑩ diagnose(原生多模态)(2 处)|

**为什么这样分**:
- flash 拿来做"翻译型" 任务(填槽、改写 query、归位回答)— pro 在这种任务上**没增量价值** 但贵 5×、慢 5-10×
- pro 留给真要"推理"的环节(鉴别诊断、安全护栏、用药方案)— 临床推理深度跟模型能力直接挂钩
- vision 专门跑要看图的(报告解析、⑩ 多模态诊断) — flash/pro 都不支持视觉

实际收益(2026-05-24 切 flash 后):② dense_query 从 85s → 15s,⑬ format_response 从 8s → 2s,info_collect 从 ~60s → ~12s。**端到端延迟整体下降 ~40%**。详见 [DEV_SPEC §9.3 LLM 调用清单](DEV_SPEC.md#9-全局实现契约跨章节)。

### 7. 鲁棒性:写入幂等 + 运行时降级

承认基础设施会挂,在两个时间窗口都设计明确的容错路径:

**摄取期写入幂等**(PG → Milvus 串行,无共享事务协调器):
- PG 写失败 → 整批回滚,无脏数据
- Milvus 写失败 → PG 标 `failed`,补偿任务定期扫描差集重写
- `chunk_id` 由 `SHA256(source_id : heading_path_id : relative_chunk_index)` 生成,**所有 upsert 全程幂等**

**运行时降级**(基础设施抖动不连累业务):
- **Redis 挂** → 配置缓存回源 PostgreSQL,`/readyz` 仍 ready
- **Redis 限流挂** → fail-open 放行(限流是 best-effort 防滥用)
- **Reranker 超时** → fallback 召回原序(降级排序不致命)

不写无止境重试,**故障半径控死在一层**。详见 [DEV_SPEC §3.1.4 / §3.1.6](DEV_SPEC.md#314-幂等性设计idempotency)。

### 8. 12 维 HPI 结构化主动问诊

不是被动等用户自由描述,按医学**现病史 12 维 schema**(`PresentIllnessSlots`)主动追问:

| 维度 | 字段 | 维度 | 字段 |
|---|---|---|---|
| 起病时间 | `onset_time` | 加重因素 | `aggravating` |
| 起病方式 | `onset_mode` | 缓解因素 | `relieving` |
| 诱因 | `trigger` | 伴随症状 | `associated_symptoms` |
| 部位 | `location` | 病程演变 | `progression` |
| 性质 | `nature` | 诊疗经过 | `treatments`(半结构化 `<治疗>: <反应>`)|
| 程度 | `severity` | 时间规律 | `duration_pattern` |

**1 次 LLM 同时出追问 + unaskable 粗筛**(节点 ⑤ `select_discriminative_symptom`):

1. **输入**:12 维 HPI 已填 / 空缺 + 已确认/否认/不确定症状 + 主诉 + 现病史
2. **LLM 任务 1 — `askable`**(≤ `MAX_FOLLOWUP_QUESTIONS=5`,可为 0):
   - `type="targeted"`:从空缺维度里挑诊断价值最高且 patient-answerable 的(时间/部位/性质/诱因/缓解等)
   - `type="open"`:开放式问"还有别的不舒服吗?"(一轮最多 1 条)
   - 信息已足时返空 → `should_continue` 路由跳诊断
3. **LLM 任务 2 — `unaskable`**(≤ 5,可为 0):"想知道但患者答不上的体征"粗筛(`{description, reason}`),后续 ⑩ 基于诊断结果精筛覆盖
4. **L2 硬去重**:每轮 askable 注入 `asked_targets` 已问列表,LLM 不能换皮重出
5. **追问硬上限**:`MAX_FOLLOWUP_ROUNDS=8` 兜底,触顶 → ⑩ Step -1 短路出 `insufficient`

> 原"TF-IDF + 信息增益 + 可问性评估"4 步算法(原 `extract_symptoms` + 4 LLM)整体废弃 — 实测 TF-IDF 抽出 94% 是医学教材通用高频词,信息增益的可比 key 立不起来。改 LLM 1 次基于 state 直接选,利用 LLM 内化的医学鉴别诊断知识。详见 [EL_DESIGN_REVIEW §11](docs/EL_DESIGN_REVIEW.md)。

### 9. 1 步诊断推理 + 失败兜底

⑩ `diagnose` 1 LLM 调用直接出 `DiagnosisOutput`(原生多模态 DashScope qwen3.5-plus,对齐 RAG 评测口径):

- **输入**:全量患者画像(主诉 + 现病史 + 12 维 slots + 已确认/否认/不确定症状 + 病史摘要 + 报告发现)+ 文献父块全文 + figure 多模态截图 + ⑤ 粗筛 unaskable
- **输出**:`results: list[RankedDisease]`(每项含 `disease/probability/evidence/differentiation/differentiation_type`)+ `retained_unaskable`(基于诊断结果精筛后覆盖 state,供 ⑧a 消费)
- **失败兜底**:LLM 重试 3 次仍失败 → 出 `insufficient` 并写 `failure_reason="step_1_structured_output_failed: ..."`;还有 Step -1 在 `followup_round >= MAX_FOLLOWUP_ROUNDS` 时直接短路

> 原"3 步链(`EvidenceSheet → DiagnosisRanking → DiagnosisOutput`)"整体废弃 — 评测证明 1 步 LLM + 信息全给已经能拿 top1 93.5% / top2 100%,3 步链让总延迟 4-6 分钟、且 Step 3"概率校准"是伪能力(同款 LLM 自校自不会本质改变判断)。详见 [DEV_SPEC §4.1.2 ⑩](DEV_SPEC.md#41-agent工作流) + [§9.1 / §9.3](DEV_SPEC.md#9-全局实现契约跨章节)。

### 10. Safety Gate 作为硬性安全闸

⑪ `safety_gate` 不是事后检查,而是 ⑩ → ⑫ 之间的硬性闸口:

- 规则层:`allergy_history` 直接抽取过敏原 → 禁用同类药
- LLM 兜底:判断交叉过敏、妊娠/哺乳禁忌、药物相互作用
- 失败保守追加通用警告 — **绝不让 ⑫ 在缺少 `safety_constraints` 的情况下生成用药建议**

### 11. 全链路审计 — `rag_trace` 15 字段

每轮诊断在 PG 写入一行 `rag_trace`(15 字段含检索 / 重排 / 失败原因 / token / 各阶段延时,JSONB,90 天保留,GIN 索引按 chunk_id 反查影响面)。

实现风格刻意没有装饰器 / helper,每个调用点裸写 try/except + 6 个 Prometheus 指标手动 `.inc()` / `.observe()` — 让重试与失败路径在代码里**显式可见**。详见 [DEV_SPEC §9.1 / §9.6](DEV_SPEC.md#9-全局实现契约跨章节)。

### 12. 运行时常量集中管理(§9.7 `agent_limits`)

7 个阈值与硬上限(`MAX_FOLLOWUP_ROUNDS=8` / `MAX_EXAM_ROUNDS=3` / `MAX_FOLLOWUP_QUESTIONS=5` / `RETRIEVE_TOP_N=200` / `RERANKER_CUTOFF_LAYERS=None` 全 40 层 / `RETRIEVE_PARENT_FIGURE_CAP=5` 同节图表封顶 / `RRF_DENSE_WEIGHT_FACTOR=5` 加权融合)统一在 `pydantic_settings.BaseSettings`,环境变量前缀 `AGENT_*` 可调,业务代码必须 `from config.settings import settings` 读取,**禁止模块级 `MAX_X = 8`**。调参改 `.env` 不改代码。完整清单见 [DEV_SPEC §9.7](DEV_SPEC.md#9-全局实现契约跨章节)。

---

## 系统架构

> 引自 [DEV_SPEC §1.3.3](DEV_SPEC.md#133-项目层级)。

```mermaid
graph TB
    subgraph 外部
        Client["外部请求 / 浏览器"]
    end

    subgraph 云端["云端 LLM"]
        cloudapi["DeepSeek (主链)<br/><i>OpenAI-compatible</i><br/>DashScope qwen3.5-plus (多模态)<br/><i>F2.5 / F9 报告解析</i>"]
    end

    Client -->|"HTTP 80/443"| nginx

    subgraph 主链路
        direction LR
        nginx["nginx<br/><i>反向代理 · nginx:alpine</i><br/><i>配置: infra/docker/nginx.conf</i>"]
        nginx -->|"HTTP 8000"| api

        subgraph api["容器: api &ensp;Dockerfile.api"]
            direction LR
            L1["FastAPI + Auth/RateLimit<br/>Middleware + API Routes"]
            L2["LangGraph Agent<br/>（graph.py / nodes / routers）"]
            L3["RAG Pipeline<br/>（ingestion / retrieval / context）"]
            L4["Qwen3-Embedding-8B（GPU）<br/>+ BGE-Reranker-v2-minicpm-layerwise（GPU）"]
            L1 --- L2 --- L3 --- L4
        end
    end

    subgraph 数据层
        direction LR
        subgraph Milvus["Milvus 容器组（3个）"]
            direction LR
            standalone["standalone :19530<br/><i>Dense + Sparse</i>"]
            etcd["etcd<br/><i>元数据</i>"]
            minio["minio :9000<br/><i>对象存储</i>"]
        end
        postgres["postgres<br/><i>20 表 · Chunk/术语/患者 · 原始文档(MinerU 产物) · 审计</i>"]
        redis["redis<br/><i>缓存 · 仅 system_config / 60s TTL</i>"]
    end

    api -->|"HTTPS"| cloudapi
    api -->|"TCP 19530"| standalone
    api -->|"TCP 5432"| postgres
    api -->|"TCP 6379"| redis

    subgraph 监控层["监控容器组（独立，故障不影响主服务）"]
        direction TB
        subgraph 监控行1[" "]
            direction LR
            prometheus["prometheus :9090"]
            grafana["grafana :3000"]
            loki["loki :3100"]
            prometheus ~~~ grafana ~~~ loki
        end
        subgraph 监控行2[" "]
            direction LR
            promtail["promtail<br/><i>日志采集</i>"]
            node_exp["node-exporter :9100<br/><i>宿主机</i>"]
            dcgm_exp["dcgm-exporter :9400<br/><i>GPU</i>"]
            promtail ~~~ node_exp ~~~ dcgm_exp
        end
        监控行1 ~~~ 监控行2
    end

    style api fill:#e8f4fd,stroke:#2196F3,color:#1a1a1a
    style Milvus fill:#fff3e0,stroke:#FF9800,color:#1a1a1a
    style 监控层 fill:#f3e5f5,stroke:#9C27B0,color:#1a1a1a
    style 云端 fill:#e8f5e9,stroke:#4CAF50,color:#1a1a1a
    style 监控行1 fill:transparent,stroke:none
    style 监控行2 fill:transparent,stroke:none
```

---

## Agent 工作流

> 引自 [DEV_SPEC §4.1.3](DEV_SPEC.md#413-edge-路由与条件分支) — 实线为正常顺序边,虚线为条件路由(`should_continue` / `diagnose_router` / `intake_router`)。

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__(["⓪ __start__"]):::first
    N0a("⓪a initial_ask<br/><i>• 0 LLM, 出 3 个问题等回答:还有别的不舒服吗 / 过敏慢病用药 / 孕期哺乳(女性才问)<br/>• 拉患者档案</i>")
    N1("① info_collect<br/><i>一次 LLM 同时:<br/>• 拆主诉 + 现病史 + 12 维细节<br/>• 解析 ⓪a 答的过敏/慢病/孕期 + 提取新症状, 合并写回 state</i>")
    N1b("①.5 analyze_initial_reports<br/><i>interrupt 问有无报告 → 加载/多模态解析 → report_findings</i>")
    N1c("intake_followup_ask<br/><i>• 分批问 12 维slot(每批 4 个 + 1 个'还有别的不适吗')<br/>• 全部收完一次 LLM 综合翻译,归位回答至结构化字段 </i>")
    N2("② build_query<br/><i>Sparse 多字段直采 + LLM Dense Query 改写</i>")
    N3("③ retrieve<br/><i>全量向量召回</i>")
    N4("④ select_discriminative_symptom<br/><i>根据召回结果出鉴别诊断追问 + 同时写 unaskable 粗筛(供 ⑩ 精筛)</i>")
    subgraph FollowupLoop[" "]
        direction TB
        N5("⑤ generate_followup<br/><i>检索前 holistic gate(1 次 LLM 看全量 state)<br/>• askable:患者主观能答的(性质/诱因/缓解/放射/伴随…)<br/>• unaskable:必须查体/化验/影像才能知道的(Murphy 征/T3/B 超…)<br/>• unaskable_symptoms 每轮都写, 供下游 ⑩/⑧a 消费</i>")
        N6("⑥ wait_followup_answer<br/><i>等用户回答追问</i>")
        N7("⑦ process_followup_answer<br/><i>LLM 把用户回答翻译成结构化字段写回 state, 清掉本轮问题列表</i>")
    end
    N8("⑧a recommend_exam<br/><i>综合已有信息 → 推患者友好检查清单(两入口都可补常规鉴别项)<br/>• ⑤ 入口: 无诊断结果, 凭主诉判<br/>• ⑩ 入口: 有诊断结果当 prior, 按候选排优先级</i>")
    N8b("⑧b wait_exam_report<br/><i>interrupt 等待检查结果</i>")
    N9("⑨ process_exam_result<br/><i>处理检查结果回传</i>")
    N10("⑩ diagnose<br/><i>诊断推理(可选 Cross-Encoder 截断 + 父块扩展 + 多模态 LLM 一步出结果)<br/>• 同时精筛/改写/新产 unaskable_symptoms(覆盖上游粗筛, 供 ⑧a 推检查)</i>")
    N11("⑪ safety_gate<br/><i>安全约束门控（规则+LLM）</i>")
    N12("⑫ generate_advice<br/><i>生成建议</i>")
    N13("⑬ format_response<br/><i>格式化最终回复</i>")
    __end__([__end__]):::last

    __start__ -->|"单边入口"| N0a;
    N0a -->|"用户答完 → ① 一次 LLM 同时抽主诉 + 解析用户答案"| N1;
    N1 -->|"主诉/现病史 已抽出 → 看患者要不要传报告"| N1b;
    N1b -->|"报告环节走完 → 开始入站追问"| N1c;
    N1c -->|"slot 已翻译进 state, 交给 ⑤ 做检索前 holistic gate"| N5;
    N2 -->|"组装检索 query"| N3;
    N3 -->|"拿到候选医学知识 chunks"| N4;
    N4 -.->|"信息够了 → 直接诊断"| N10;
    N4 -.->|"还需追问鉴别诊断信息 → 直接出题让用户答"| N6;
    N5 -.->|"askable 问题 → 让用户答"| N6;
    N5 -.->|"无 askable 但有 unaskable → 走首诊推单"| N8;
    N5 -.->|"都空 → 直接进检索"| N2;
    N6 -->|"用户答完 → 由 ⑦ 翻译"| N7;
    N7 -.->|"还没进过检索(intake 后这条路) → 回 ⑤ 再判一次"| N5;
    N7 -.->|"已经检索过(④ 鉴别诊断追问答完), 或轮数已达上限 → 回检索"| N2;
    N10 -.->|"需要患者去做检查再鉴别"| N8;
    N10 -.->|"可下结论 → 走安全门控"| N11;
    N8 -->|"生成建议写入State"| N8b;
    N8b -->|"interrupt等待检查结果"| N9;
    N9 -->|"新证据回传,重新召回"| N2;
    N11 -->|"过敏/用药/妊娠约束过滤"| N12;
    N12 -->|"用药/检查/高危提示"| N13;
    N13 -->|"输出最终回复"| __end__;

    classDef default fill:#ffffff,stroke:#333,stroke-width:1px,color:#000,line-height:1.2;
    classDef first fill:#f0f0f0,stroke:#999,color:#000;
    classDef last fill:#d4cffc,stroke:#333,color:#000;
    style FollowupLoop fill:none,stroke:none
```

> 易错点(实现时会反复遇到):
>
> - ⑧/⑥ 拆 `a`/`b` 两半,`interrupt()` 与 LLM 调用分离,resume 时 LLM 不会重发
> - `should_continue` 是 **纯函数**,不允许写 state(cap 处理放在 ⑩ Step -1)
> - ⑩ 1 步 LLM 失败 → 短路 `insufficient` + `failure_reason="step_1_structured_output_failed: ..."`
> - `present_illness_slots` 12 维度,空槽驱动 ⑤ 维度追问
> - ⑤ holistic gate 出 `askable_targets`,⑦ 落回 state 后 append 到 `asked_targets` 做 L2 硬去重,防 LLM 换皮重出

---

## RAG 流水线

### 摄取(Ingestion)— 总体流水线

```mermaid
flowchart TD
    PDF[PDF 教材<br/>13 本 13912 页] --> MR[MinerU<br/>VLM + pipeline 双引擎]
    MR --> RD[(raw_documents<br/>JSONB + GIN)]
    RD --> CK[Chunking<br/>目录权威清单 + 三遍切<br/>+ size 驱动子块]

    CK -->|child + parent| EN1[Enrichment<br/>title + summary + 3 questions<br/>DeepSeek 结构化输出]
    CK -->|table| EN2[Enrichment<br/>medical_statement + 4 字段<br/>DeepSeek 看 html]
    CK -->|figure / chart / flowchart| EN3[Enrichment<br/>medical_statement + 4 字段<br/>DashScope 看截图]

    EN1 --> CH[(chunks 表<br/>26054 行)]
    EN2 --> CH
    EN3 --> CH

    CH -->|child| EM[Embedding<br/>Qwen3-8B INT8 batch=8]
    CH -.parent skip.-> CH
    EM -->|1 original<br/>1 summary<br/>3 question| MV[(Milvus docs_collection<br/>129810 entities)]

    EM -.失败标 failed.-> COMP[补偿任务<br/>差集重写]
    COMP --> MV

    classDef llm fill:#fef3c7,stroke:#d97706,color:#1a1a1a
    classDef gpu fill:#dcfce7,stroke:#16a34a,color:#1a1a1a
    classDef store fill:#e0e7ff,stroke:#4338ca,color:#1a1a1a
    class EN1,EN2,EN3 llm
    class EM,MR gpu
    class RD,CH,MV store
```

### 摄取 — PG → Milvus 写入流程(放大看)

> 引自 [DEV_SPEC §3.1.6.2](DEV_SPEC.md#3162-写入流程以-batch-为单位) — `embedding_status` 字段(`pending → done / failed / skip`)兼任两阶段状态机,不需要分布式事务。

```mermaid
flowchart TD
    S1["<b>Step 1: PostgreSQL Upsert（事务）</b><br/>- Upsert chunks 表（chunk_id 为主键）<br/>- 父块：embedding_status = 'skip'<br/>- 子块：embedding_status = 'pending'<br/>- 事务提交"]
    S1 -->|成功| BRANCH
    S1 -->|"失败 → 整批回滚，不进入 Step 2，<br/>下次重试整批重处理"| END1["终止（等待重试）"]

    BRANCH{"父块？<br/>(embedding_status = 'skip')"}
    BRANCH -->|是| END2["终止（无需向量化）"]
    BRANCH -->|否（子块）| S2

    S2["<b>Step 2: Milvus Upsert（批量）</b><br/>- 以 chunk_id 确定性派生向量记录 ID：<br/>&ensp;{chunk_id} (original) / {chunk_id}_summary /<br/>&ensp;{chunk_id}_q0 / {chunk_id}_q1 / ...<br/>- 派生规则与字段定义见 config/milvus_schema.py:119 + src/rag/ingestion/embedding.py:112<br/>- 批量 Upsert 到 docs_collection"]
    S2 -->|成功| S3
    S2 -->|失败| S2a

    S2a["<b>Step 2a: Milvus 写入失败处理</b><br/>- PostgreSQL 中对应 chunk 的 embedding_status 更新为 'failed'<br/>- 记录错误日志（chunk_id + 异常信息）<br/>- 不回滚 PostgreSQL 数据（元数据本身是正确的）<br/>- 由补偿任务定期重试（见下方「补偿机制」）"]

    S3["<b>Step 3: 确认写入</b><br/>- Milvus Upsert 成功后，更新 PostgreSQL：<br/>&ensp;embedding_status = 'done', updated_at = now()"]
```

### 检索(Retrieval)

```mermaid
flowchart TD
    Q[用户 query] --> BQ[② build_query<br/>NER + Sparse 多字段直采]
    BQ -->|dense_query| DE[Dense Route<br/>Qwen Embedding<br/>1 次 ANN]
    BQ -->|sparse_queries N 维| SP[Sparse Route<br/>N 次 BM25]

    DE --> RRF[单阶段多路 RRF<br/>k=60 等权]
    SP --> RRF

    RRF --> AGG[多向量聚合<br/>by source_chunk_id<br/>同 chunk 跨向量得分求和]
    AGG --> TOP[Top-N 截断<br/>RETRIEVE_TOP_N=200]
    TOP --> EXT[④ 智能追问选择<br/>1 LLM 出 questions + unaskable 粗筛]
    EXT -.多轮追问.-> BQ

    TOP --> RR[⑩ Step 0 Reranker<br/>BGE-MiniCPM layerwise<br/>失败 → 原序 fallback]
    RR --> EXP[Context 扩展<br/>子→父 + 同节图表]
    EXP --> DG[⑩ 1 步 LLM 诊断<br/>原生多模态]

    classDef gpu fill:#dcfce7,stroke:#16a34a,color:#1a1a1a
    classDef key fill:#fef3c7,stroke:#d97706,color:#1a1a1a
    class DE,RR gpu
    class RRF,AGG,RR key
```

---

## 数据层

PostgreSQL 20 表按职责分四组,完整 SQL DDL 见 [DEV_SPEC §2.4](DEV_SPEC.md#24-数据存储选型及具体设计):

| 分组 | 表 | 用途 |
|---|---|---|
| 用户 / 会话 | `users` / `sessions` / `conversations` | 账号 + 多轮问诊 |
| 患者档案(8 表) | `patients` / `medical_history` / `surgical_trauma_history` / `transfusion_history` / `allergies` / `medications` / `family_history` / `menstrual_reproductive` / `exam_reports` | 八大采集规范 |
| 知识库 | `sources` / `raw_documents` / `chunks` | MinerU 产物 + chunk 元数据 |
| 审计 / 配置 | `rag_trace` / `kb_change_log` / `config_change_log` / `diagnosis_feedback` / `system_config` | 90 天保留 + 同事务双写 |

---

## 技术栈与选型理由

| 层 | 选型 | 选型理由(摘要,详细见 [DEV_SPEC §2](DEV_SPEC.md#2-技术选型)) |
|---|---|---|
| Embedding | Qwen3-Embedding-8B(INT8,4096 维) | 8B 参数容量足够编码医学术语细粒度差异;C-MTEB 73.84 远超 BGE-M3;与 LLM 同源 tokenizer |
| Reranker | BGE-Reranker-v2-minicpm-layerwise(INT8) | MiniCPM 中文医疗最佳;cross-encoder 全交互弥补与 Qwen 异源;layerwise 提供精度/速度旋钮 |
| LLM 主链 | DeepSeek(`with_structured_output` `function_calling`) | 廉价 + 14 处结构化调用稳定 |
| LLM 多模态 | DashScope qwen3.5-plus | 报告解析专用,主链 deepseek 不支持视觉 |
| 向量库 | Milvus 2.5 standalone | HNSW + 内置 BM25(中文 analyzer);2.4+ 全文检索一站式,无需 SPLADE |
| 关系库 | PostgreSQL 16 | JSONB + GIN 容纳 MinerU 异构产物;ACID 事务保障双写一致 |
| 缓存 | Redis 7 | 仅缓存 system_config(60s TTL);**RAG 响应不缓存**(state 跨调用变化,query 跨患者会冲撞) |
| 编排 | LangGraph | 内置 interrupt/resume + checkpointer + thread_id 隔离 |
| Web 框架 | FastAPI + Pydantic v2 + uvicorn | 异步 + 自动 OpenAPI;`prometheus-fastapi-instrumentator` 自动 HTTP 指标 |
| 鉴权 | PyJWT HS256 + bcrypt + 角色守卫 | `require_role(*roles)` 工厂模式;限流 key 走 `user:<sub>` 退化 `ip:<addr>`(伪造 token 落 IP 桶) |
| 限流 | 自研滑动窗口(`Protocol` 抽象) | H6 切 Redis 只换实现不动业务 |
| 解析 | MinerU 2.5 | VLM(MinerU2.5-Pro 1.2B)+ pipeline(PDF-Extract-Kit)双引擎 hybrid auto |
| 监控 | Prometheus + Grafana + Loki + Promtail + DCGM | 应用/系统/GPU/日志全栈 |
| 反代 | Nginx | `set_real_ip_from + X-Forwarded-For` 透传客户端 IP 让限流可用 |
| 包管理 | uv + Python 3.12 + PyTorch 2.7 cu128 | Blackwell sm_120 仅在 cu128+ wheel |
| 测试 | pytest 三层 | unit(全 mock)/ integration(真 docker)/ e2e(真 LLM,J 阶段) |

---

## 监控与可观测性

13 容器一键启动后,Prometheus 自动 scrape 6 目标(api / postgres / redis / milvus / node / dcgm),Grafana 启动自动加载 2 仪表盘。所有 LLM 调用埋点**完全裸写** —— 无装饰器 / 无 helper / 无 context manager(详见 [DEV_SPEC §9.1](DEV_SPEC.md#9-全局实现契约跨章节)),20+ 调用点约 300 行重复样板换"异常作用域清晰 + 重试可观察 + 与 `with_retry` 内部行为不打架"。

<div align="center">
  <img src="assets/grafana-app-performance.png" alt="Grafana — 应用性能大盘" width="720"/>
  <br/>
  <sub>应用性能:QPS / 5xx / HTTP P95 / LLM 重型(pro·vision)轻型(flash) 延迟 / 重试·兜底 / 失败分布 / PG·Redis·Milvus 依赖层</sub>
  <br/><br/>
  <img src="assets/grafana-hardware-resources.png" alt="Grafana — 硬件资源大盘" width="720"/>
  <br/>
  <sub>硬件资源:GPU 利用率 / 显存占用 / CPU / 内存 / 磁盘 / 网络(DCGM + Node Exporter)</sub>
</div>

---

## 快速开始

主路径 = 全 docker compose 一键起。host 跑 uvicorn 仅作为热重载调试备选。

```bash
# 1. 克隆
git clone https://github.com/Handshakeworm/Agentic-RAG-diagnose-Assistant.git
cd Agentic-RAG-diagnose-Assistant

# 2. 配置环境变量
cp .env.example .env
# 必填:
#   LLM_API_KEY            (DeepSeek)
#   LLM_VISION_API_KEY     (DashScope)
#   JWT_SECRET_KEY         (openssl rand -hex 32)

# 3. 起全栈(13 容器:nginx + api + 数据层 + 监控)
#    首次 build api 镜像 ~10-15 分钟(下 PyTorch + nvidia-* wheels ~5GB)
#    需 nvidia-container-toolkit(GPU 透传给 api / dcgm-exporter)
docker compose up -d --build
docker compose ps                            # 全 healthy 才算就绪

# 4. 初始化数据库 schema(20 张表 + 索引,首次部署一次)
docker compose exec api alembic upgrade head
```

**验证**(经 nginx 80):

```bash
curl http://localhost/healthz                # {"status":"ok"}
curl -i http://localhost/readyz              # 200 + x-trace-id header
curl http://localhost/metrics | head         # Prometheus 业务指标
curl -X POST http://localhost/auth/register \
  -H 'content-type: application/json' \
  -d '{"email":"a@b.com","password":"hunter22","role":"patient"}'
# → {"access_token":"...", ...}
```

打开 `http://localhost:3000`(admin/admin)→ Dashboards → "Medical RAG" → "应用性能" / "硬件资源" 实时大盘。

**热重载开发**(改 src/ 想立刻看效果时):

```bash
# 主 compose 起依赖,api 在 host 跑(可热重载 + IDE debugger)
docker compose up -d postgres redis milvus-standalone prometheus loki grafana promtail node-exporter dcgm-exporter
uv sync
source .venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 想用 nginx 反代到 host uvicorn(让限流 / trace_id 链路也走通):
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d nginx
```

## 评测结果

**评测数据**:62 case 执业医考题(单主诊断 + 教科书初诊答案),全量入链路不预筛。

**两层评测**:① 检索质量(RAG 召回是否覆盖正确章节) + ② 诊断准确率(LLM 基于召回 chunk 能否得出正确诊断)。

### 1. 检索质量(parent-level,62 case)

LLM Judge(DeepSeek)对 RRF 加权 Top-50 parents 评 0~3 分,得到 ground truth 排序,再对照 prediction 算 NDCG / Spearman。

| 指标 | parent-level | 解读 |
|---|---|---|
| **NDCG@20** | **0.774** | 排序质量高(医疗专域 0.6~0.8) |
| **Hit@20(≥3)** | **100%** | **0 case 漏掉高相关章节** |
| **MRR(≥2)** | **0.954** | 几乎所有 case Top-1 即命中中相关 |
| **Spearman ρ vs LLM** | **+0.708** | 排序高度对齐 LLM 判断 |
| Avg score @20 | 1.944 | Top-20 平均 14 个中度以上相关 |

### 2. 诊断准确率(62 case,完整 RAG → ⑩ 闭环)

把 RAG 加权 Top-20 parents + 病例信息(含 figure 截图)喂 vision LLM(qwen3.5-plus),输出候选疾病列表 + 概率;再用 LLM Judge(DeepSeek)按"临床本质等价"评判 gold vs LLM Top-K 命中。

> **术语说明**:
> - **"gold"** = 教科书答案里的初步诊断 list(从执业医考题 PDF 抽取)
> - **"主诊断" / "gold[0]"** = list 的第一项,通常是病例主要疾病(如"急性心肌梗死")
> - **"多 gold / 多诊断 case"** = 同一患者**同时存在多个并存疾病/合并症**的 case(如 case 014 "脾破裂 + 肋骨骨折"、case 028 "Graves 病 + 甲亢性心脏病"、case 043 "冠心病急性心梗 + 急性左心衰竭" — 62 case 中约一半属此类)
> - **multi-gold recall** = 多 gold case 平均**几个 gold 被 LLM 候选覆盖**的比例;**全 gold 命中** = 所有 gold 都被 LLM 给出对应 candidate

| 指标 | 数值 | 解读 |
|---|---|---|
| **Top1 命中率(loose)** | **93.5%** | 主诊断在 rank=1 临床等价(58/62) |
| **Top1 命中率(strict)** | 61.3% | rank=1 且 exact/equivalent/more_specific(38/62) |
| **Top2 命中率** | **100% 🎯** | gold 主诊断全部在 top-2(62/62)— 4 个非 top1 命中 case 全部排 rank=2 |
| **Top5 命中率** | 100% | |
| **MRR(loose)** | ~0.96 | 几乎所有 case top1 就给出正确诊断 |
| **Multi-gold recall mean** | 0.867 | 多并发诊断 case 平均 86.7% gold 被覆盖 |
| **全 gold 命中 case** | 45/62 (72.6%) | 多并发诊断 case 所有 gold 都被 LLM 列出 |
| **0 主诊断 none** | 0/62 | 没有任何 case 主诊断方向错 |

**match_type 分布(主诊断)**:

| 类型 | 数量 | 含义 |
|---|---|---|
| exact | 9 | 字面完全相同 |
| **equivalent** | **17** | 同病不同表述(Graves 病 = 毒性弥漫性甲状腺肿)|
| **more_specific** | **15** | LLM 比 gold 更精确(STEMI vs 急性心梗) |
| more_general | 13 | LLM 比 gold 笼统 |
| partial | 8 | 多 gold 只命中部分 |
| none | 0 | 完全不匹配 |

### 3. 关键设计决策(均经评测验证)

| 决策 | 评测依据 |
|---|---|
| **Reranker 默认关闭** | K=20 下 BGE Reranker 全主指标无优势(NDCG -0.076 / Hit -1.6pp / MRR -0.065),省 2.6GB GPU + 5s 延迟 |
| **RRF 动态加权** `dense_weight = max(1, N_sparse/5)` | 等权下 dense 单路被 sparse 多路挤兑,加权后 Top-20 内 dense exclusive chunks 保留量 ×6.5 |
| **Sparse 多字段直采** | state 字段拼接(chief + 12 维 slots + report findings)→ N 路独立 BM25;实测 N=12~30(均 21.8),信号比单 query 拼接更全 |
| **20 unique parents 喂 LLM**(对齐 LLM Judge 口径) | parent 不参与 embedding,需从 Top-200 chunks 顺序去重 |

### 4. 评测方法 + 数据 + 脚本

- **评测数据**:`.eval/rag_eval/cases/*.json` — 62 case 执业医考题(`patient_text` + `diagnosis` gold)
- **检索评测**:`.eval/rag_eval/{compare_rrf_weighting,run_llm_judge,compute_metrics}.py`
- **诊断评测**:`.eval/rag_eval/{run_diagnose_eval,run_diagnose_judge}.py`(LLM Judge 评 gold ↔ LLM 候选临床等价性,6 档 match_type)
- **RAGAS 4 指标评测**:`.eval/rag_eval/run_ragas_judge.py` — Faithfulness / Answer Relevancy / Context Precision / Context Recall,**对齐 ragas v0.2 原算法**(自实现 LLM Judge + 项目 Qwen3-Embedding-8B 算 cosine,不依赖 ragas 库;选型理由见下方"详细方法学")
- **per-case 结果**:
  - 诊断结果 `.eval/rag_eval/diagnose_eval/*.json`(62 个 + gold + LLM candidates + prompt + raw)
  - 诊断 LLM Judge `.eval/rag_eval/diagnose_judge/*.json` + `diagnose_judge_summary.json`
  - RAGAS LLM Judge `.eval/rag_eval/ragas_judge/*.json` + `ragas_summary.json`
- **详细方法学**:
  - 检索层 — [RETRIEVAL_EVAL.md](docs/RETRIEVAL_EVAL.md)(7 章,含 Reranker 全 K 对照、双粒度指标)
  - 诊断层 + RAGAS — [EVALUATION_METHODOLOGY.md](docs/EVALUATION_METHODOLOGY.md)(诊断 Judge 评判标准 6 档 + RAGAS 4 指标算法详解 + 跟 ragas 库的对齐说明 + 简化 / 等价点逐条标注)

### 5. 真实 case 示例 — 完整 LLM 诊断输出

以 case 001 为例(典型 EDH 病例,Top-1 命中):

> **病例**(节选):男性 23 岁,骑车撞倒右颞部着地半小时。曾昏迷 5 分钟,清醒后头痛恶心。**2 小时后头痛加重 + 呕吐 + 烦躁 + 浅昏迷,右瞳散大对光反射迟钝,左 Babinski 阳性**。头颅平片:右额颞线形骨折。
>
> **Gold(教科书初诊)**:右额颞急性硬膜外血肿

LLM 给出的 4 个候选(按概率降序):

| Rank | 疾病 | Prob | 关键证据(LLM 自己写) | LLM Judge 评判 |
|---|---|---|---|---|
| **1** | **右颞部急性硬脑膜外血肿** | **0.95** | "典型'中间清醒期'病史:伤后昏迷 5 min,清醒后头痛恶心,2h 后再次意识障碍" + "右颞部着地 + 右额颞线形骨折,易损伤脑膜中动脉导致 EDH" | ✅ **equivalent**(临床本质同一疾病,部位细微差异)|
| 2 | 右颞部急性硬脑膜下血肿 | 0.35 | "颅压增高 + 脑疝体征与本例相符" | 鉴别诊断:"SDH 多由桥静脉撕裂,中间清醒期较少见,CT 月牙形 vs EDH 双凸形" |
| 3 | 右颞叶脑挫裂伤伴脑内血肿 | 0.25 | "右颞部直接受力 + 2h 病情恶化可能为迟发性出血" | 鉴别诊断 |
| 4 | 弥漫性轴索损伤(DAI) | 0.05 | "车祸旋转加速机制" | 鉴别诊断:"DAI 立即持续昏迷,无中间清醒期,本例不支持" |

**亮点**:LLM 不仅给出主诊断,还**自主完成了完整的临床鉴别诊断推理**(EDH vs SDH vs 脑内血肿 vs DAI),引用召回的文献章节(`文本块 N`)作证据,并给出 CT 特征等可操作的鉴别要点 — 接近临床医生的初诊思维过程。

### 6. 评测局限(诚实展示)

- **数据是执业医考题**(单主诊断为主 / 信息相对完整),**真实临床多病并发场景**复杂度更高;主诊断 100% 入 top-2,但多 gold case 的次诊断/合并症 LLM 不一定独立列出(case 062 麻疹给了"麻疹"但没单列"合并肺炎"作 candidate)
- **评测为单轮**(`patient_text` 一次性输入),没跑 Agent 多轮追问;production 真实场景会有 ④/⑤/⑥/⑦ 追问循环产出 `confirmed_symptoms` / `medical_history` 信息,目前评测里这些字段空
- **gold 是教科书"初步诊断"答案**,带"可能性大" / "待除外" / "初步诊断:" 等不确定语气;Judge 已按"初步诊断"口径校准(忽略这类语气词)
- **LLM Judge 评等价性**:DeepSeek 评 gold ↔ LLM 等价等级,本身有 LLM bias 风险;case-by-case 人工抽查显示评判合理(详见 [.eval/rag_eval/diagnose_judge/](.eval/rag_eval/diagnose_judge/))
- **真实"误诊"** 只 1 例 — case 049 "慢性菌痢",LLM top1 给 "溃疡性结肠炎"(gold 在 top2 equivalent 命中);1 例 case 007 输尿管结石 LLM 一开始给 "肿瘤" 是合理临床歧义(B 超未见结石强回声 + 55 岁吸烟 → 倾向肿瘤),重跑后给出结石

---

## 项目进度

详见 [DEV_SPEC.md §8.4 进度跟踪表](DEV_SPEC.md#84-进度跟踪表)。截至 2026-05-17:

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | 工程骨架与基础设施基座 | 完成 |
| B | 数据层与模型客户端 | 完成 |
| C | Ingestion Pipeline | 主流程跑通(13 本灌库),production 化收口待补 |
| D | 术语库(数据备用) | ICD-10 alias 40k+ 已灌库,运行时不调用 |
| E | Retrieval(Sparse / Dense / RRF / Reranker / Filter) | 完成 |
| F | Agent 工作流(17 节点 + 4 条件路由) | 完成 |
| G | API 层与权限系统(7 项) | 完成 |
| H | 基础设施增强(Redis / Prometheus / Grafana / Loki / DCGM) | 完成 |
| I | 评估体系 | **主体完成**(检索 RAG 评测 + 诊断闭环 + 双层 LLM Judge,见上"评测结果";Agent 多轮追问评测待补)|
| J | 端到端验收与文档收口 | J0(Docker 化部署)完成,J1-J6 待开始 |

**测试覆盖**:unit 333 PASS / integration 95 PASS(真 PG + Milvus + Redis)/ e2e 留 J1-J4;skip 17(GPU 模型 + 已知 Milvus race)。

---

## 目录结构

```
.
├── DEV_SPEC.md              # 4976 行设计规范(单一事实源)
├── CLAUDE.md                # AI 协作工作流与契约红线
├── pyproject.toml           # uv 依赖,锁定 cu128 wheel index
├── docker-compose.yml       # 数据层 + 监控基础设施(13 容器)
├── docker-compose.demo.yml  # ⚠️ DEPRECATED 自 J0,仅 host 热重载调试用
├── .dockerignore            # docker build 上下文排除(J0)
├── alembic.ini              # 数据库迁移配置
│
├── src/
│   ├── agent/               # LangGraph 17 节点 + state.py + graph.py
│   │   ├── nodes/           # ① info_collect ~ ⑬ format_response
│   │   ├── routers/         # should_continue / diagnose_router
│   │   └── utils/           # patient_repo / report_parser / chunks_lookup
│   ├── rag/
│   │   ├── ingestion/       # MinerU loader / chunking / enrichment / embedding / idempotency
│   │   └── retrieval/       # sparse / dense / fusion / reranker / metadata_filter
│   ├── api/                 # FastAPI app + routes + middleware
│   │   ├── routes/          # auth / diagnosis / patient / admin
│   │   ├── middleware/      # auth_middleware (JWT) + rate_limiter (滑动窗口)
│   │   └── schemas/         # Pydantic request/response
│   ├── db/
│   │   ├── milvus/          # docs_collection + terms_collection(数据备用)
│   │   └── postgres/        # ORM(20 表)+ Alembic versions/
│   ├── models/              # Embedding / Reranker / LLM 客户端封装
│   ├── prompts/             # ingestion + agent 12 prompt builders
│   └── common/              # metrics(§9.1)/ hashing / normalize
│
├── config/
│   ├── settings.py          # pydantic-settings(含 §9.7 agent_limits)
│   ├── milvus_schema.py     # 双 collection schema
│   └── logging_config.py
│
├── terms/                   # ICD-10 术语库构建(40k 条)
├── scripts/                 # MinerU 批解析 / chunking POC / DB 灌库
│   ├── METHODOLOGY.md       # Chunking POC 通用方法论
│   ├── batch_parse_pdfs.sh
│   ├── enrichment.py
│   ├── load_*.py            # PG / Milvus 灌库
│   └── poc_chunking_*/      # 12 本书 POC 切分结果
│
├── infra/
│   ├── docker/              # nginx.conf
│   ├── prometheus/
│   ├── loki/
│   └── promtail/
│
├── .eval/rag_eval/          # I 阶段评测数据 + 脚本(详见"评测结果")
│   ├── cases/               # 62 case 执业医考题(patient_text + gold)
│   ├── step0_cache/         # case 文本拆解为 MedicalState 字段
│   ├── sparse_fusion_compare/  # 检索评测中间产物(64MB,gitignored)
│   ├── diagnose_eval/       # 62 case 诊断完整输出
│   ├── diagnose_judge/      # LLM Judge 评判结果
│   ├── run_diagnose_eval.py / run_diagnose_judge.py
│   ├── compare_rrf_weighting.py / run_llm_judge.py / compute_metrics.py
│   ├── compare_rerank_full.py / validate_table_recall.py / step1.py
│   └── parse_pdf_to_cases.py
│
├── evaluation/              # 预留(在线追踪/生产评测,待补)
└── tests/
    ├── unit/                # 333 PASS(全 mock)
    ├── integration/         # 95 PASS(真 PG + Milvus + Redis)
    └── e2e/                 # 真 LLM API(J 阶段)
```

---

## 文档导航

- [DEV_SPEC.md](DEV_SPEC.md) — 完整设计规范
  - [§1 项目总览](DEV_SPEC.md#1-项目总览)
  - [§2 技术选型](DEV_SPEC.md#2-技术选型)(Embedding / Reranker / LLM / 数据存储的 why)
  - [§3 RAG 系统设计](DEV_SPEC.md#3-rag系统设计)(摄取 / chunking / enrichment / 多向量 / 召回 / 精排)
  - [§4 Agent 设计](DEV_SPEC.md#4-agent设计)(State / 17 节点详解 / 上下文管理)
  - [§5 基础设施](DEV_SPEC.md#5-基础设施)(性能 / 监控 / 管理)
  - [§6 评估](DEV_SPEC.md#6-评估)(RAG 离线 / Agent 离线 / LLM Judge / 在线)
  - [§9 全局实现契约](DEV_SPEC.md#9-全局实现契约跨章节) — **跨章节工程契约**(LLM 调用模板 / Pydantic schema / 运行时常量 / 审计字段)
- [CLAUDE.md](CLAUDE.md) — AI 协作工作流、架构规约、契约红线
- [RETRIEVAL_EVAL.md](docs/RETRIEVAL_EVAL.md) — RAG 检索层评测报告(7 章:sparse 字段、RRF 加权、LLM Judge、双粒度指标、Reranker 全 K 对照)
- [EVALUATION_METHODOLOGY.md](docs/EVALUATION_METHODOLOGY.md) — 诊断层 + RAGAS 4 指标评测方法学(LLM Judge 6 档评判 / RAGAS 对齐细节 / ragas 库选型理由)
- [MANUAL_TEST_CASE.md](docs/MANUAL_TEST_CASE.md) — UI 手工测试用例集
- [scripts/METHODOLOGY.md](scripts/METHODOLOGY.md) — Chunking POC 通用方法论

---

## 开发约定

1. **写代码前先定位 spec 章节**,代码 docstring 引用 `DEV_SPEC §X.Y`。无 spec 章节先与用户确认,不要凭空发明设计。
2. **触碰 LLM 调用 / Schema / 运行时常量 / 审计写入**时,先扫 §9 全局契约。详见 [CLAUDE.md](CLAUDE.md) "Sweep §9 before implementing"。
3. **修 spec 后跑** `python .claude/skills/auto-coder/scripts/sync_spec.py` 同步章节镜像 `references/`(供 Claude 按章节加载,避免一次塞 4976 行)。**不要直接编辑** `references/*.md`,会被覆盖。
4. **添加依赖改 `pyproject.toml`** + `uv sync`,不要直接 `uv pip install`(环境不可复现)。
5. **常量集中**:业务代码 `from config.settings import settings` 读 `settings.agent_limits.X`,**禁止** `MAX_X = 8` 模块级常量。
6. **测试位置**:所有 `test_*.py` 必须在 `tests/{unit,integration,e2e}/` 下,不要散落到 `scripts/`。重资源测试用 `pytestmark = pytest.mark.skipif(...)` 不要移出 tests。

---

## License

[MIT](LICENSE)
