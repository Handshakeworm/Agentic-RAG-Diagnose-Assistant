# 8. 项目排期

## 8.1 排期原则

严格对齐本 DEV_SPEC 的架构分层与 1.3.1 节目录结构。

1. **只按本文档设计落地**：以 1.3.1 节目录树为"交付清单"，每一步都要在文件系统上产生可见变化。
2. **先打通主闭环，再逐层增强**：优先做"可跑通的端到端路径（Ingestion → Retrieval → Agent 最小诊断链路 ①→①.5→②→③→④→⑩→⑪→⑫→⑬）"，再接入追问循环（⑤⑥⑦）和检查循环（⑧⑨）。
3. **外部依赖可替换/可 Mock**：LLM（Qwen）/ Embedding（Qwen3-Embedding-8B）/ Reranker / Milvus / PostgreSQL 的真实调用在单元测试中一律用 Fake/Mock，集成测试再开真实后端。
4. **每个小阶段给出验收标准**：明确"完成"的定义，避免模糊交付。
5. **基础设施按需引入**：Docker Compose 在阶段 A 搭建基座，监控/缓存等在主链路跑通后再逐步接入。

## 8.2 阶段总览

| 阶段 | 名称 | 目的 |
|------|------|------|
| **A** | 工程骨架与基础设施基座 | 建立可运行、可配置、可测试的工程骨架；Docker Compose 拉起全部存储依赖 |
| **B** | 数据层与模型客户端 | 打通 PostgreSQL / Milvus 连接；封装 Qwen3-Embedding-8B、Reranker、Qwen LLM 推理客户端 |
| **C** | Ingestion Pipeline（MinerU → Chunk → Embedding → 存储） | 离线摄取链路跑通，样例文档写入 Milvus + PostgreSQL（含 raw_documents 表存 MinerU 产物），支持幂等与增量 |
| **D** | 术语库(EL 已下线,数据备用) | 构建 terms_collection,导入 ICD-10-CN 标准术语数据(原服务 Retrieval 术语扩展和 Agent 症状预处理;**EL 整层下线后运行时不再被查询**,数据保留作为未来重新启用 EL 或切换到 EL_DESIGN_REVIEW §11.6 方案 A/B 时的基础) |
| **E** | Retrieval（Dense + Sparse + RRF + Rerank） | 在线查询链路跑通，得到 Top-K chunks（含引用信息），具备稳定回退策略 |
| **F** | Agent 工作流（LangGraph StateGraph） | 按 4.1 节设计落地 15 节点 + 2 条件路由，实现基于信息增益收敛的迭代式诊断工作流 |
| **G** | API 层与权限系统 | FastAPI 入口服务、JWT 认证、角色权限、限流，暴露问诊接口 |
| **H** | 基础设施增强（监控、缓存、日志） | Prometheus + Grafana 指标监控，Loki 日志采集，Redis 缓存客户端与缓存层 |
| **I** | 评估体系 | 离线评估（RAG + Agent）、在线追踪、LLM Judge |
| **J** | 端到端验收与文档收口 | 真实环境（非 Mock）全链路 E2E 冒烟测试，README 完善，确保开箱即用 |

## 8.3 详细排期

---

### 阶段 A：工程骨架与基础设施基座

**目的**：建立可运行、可配置、可测试的工程骨架；Docker Compose 拉起全部存储依赖（PostgreSQL、Milvus、Redis），后续所有模块都能以 TDD 方式落地。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| A1 | 初始化目录树与最小可运行入口 | 1.3.1 节完整目录结构、`pyproject.toml`、`src/__init__.py` 等 | `python -m src` 不报错；目录结构与 1.3.1 节一致 |
| A2 | Docker Compose 搭建存储基座 | `docker-compose.yml`、`infra/docker/` | `docker compose up -d` 可拉起 PostgreSQL + Milvus + Redis，各服务健康检查通过 |
| A3 | 配置加载与校验 | `config/settings.py`、`.env.example` | 从 `.env` 加载配置,缺失必填项时抛明确错误;**必须包含 §9.7 定义的 `AgentLimitsSettings` 段**(5 个常量:`MAX_FOLLOWUP_ROUNDS` / `MAX_EXAM_ROUNDS` / `MAX_FOLLOWUP_QUESTIONS` / `RETRIEVE_TOP_N` / `RERANKER_CUTOFF_LAYERS`),以 `settings.agent_limits` 嵌套属性暴露;单元测试:默认值与 §9.7 初始值一致;`.env` 覆盖 `AGENT_MAX_FOLLOWUP_ROUNDS=10` 能生效;缺失 LLM API_KEY 等必填项报错 |
| A4 | pytest 测试基座 | `tests/`、`pyproject.toml [tool.pytest]` | `pytest` 可运行，冒烟测试通过 |
| A5 | 公共工具模块 | `src/common/normalize.py`、`hashing.py`、`metrics.py` | normalize + SHA256 哈希函数单元测试通过；与 3.1.4.2 定义一致 |
| A6 | Prompt 模板骨架 | `src/prompts/ingestion.py`、`agent.py`、`evaluation.py` | 三个模块均可 `from src.prompts.xxx import yyy` 导入；函数签名与第 7 节设计一致；具体 prompt 内容随对应业务阶段（C/F/I）落地时填充。查询处理 prompt 统一归入 `agent.py` 的 `build_query_construction_prompt`，不再单列 `retrieval.py` |

---

### 阶段 B：数据层与模型客户端

**目的**：打通两层存储连接（PostgreSQL / Milvus），封装三个模型推理客户端（Qwen3-Embedding-8B / Reranker / Qwen LLM），使上层业务代码可通过统一接口调用。Redis 缓存客户端延至阶段 H 与缓存业务对接一并落地。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| B1 | PostgreSQL 连接池 + ORM 模型 | `src/db/postgres/connection.py`、`models.py` | 连接池可用；sources / raw_documents / chunks / users / patients / conversations 等表 ORM 模型与 2.4 定义一致 |
| B2 | PostgreSQL 迁移脚本 | `src/db/postgres/migrations/` | Alembic `upgrade head` 可创建全部表结构与索引（与 2.4.2、2.4.3、2.4.5 一致） |
| B3 | Milvus 连接管理 + docs_collection | `src/db/milvus/connection.py`、`docs_collection.py`、`config/milvus_schema.py` | Collection Schema 与 2.4.1 定义一致；upsert + search 集成测试通过 |
| B4 | Milvus terms_collection | `src/db/milvus/terms_collection.py` | Schema 与 2.4.6 一致；upsert + 向量检索集成测试通过 |
| B5 | PostgreSQL raw_documents 表（MinerU 产物存储） | `src/db/postgres/models.py`（raw_documents ORM 类）、`src/db/postgres/migrations/0001_raw_documents.sql` | raw_documents 表结构与 2.4.4 一致（jsonb + text + GIN 索引）；上游与 sources 表 1:1 外键约束生效；upsert 集成测试通过 |
| B6 | Qwen3-Embedding-8B 客户端 | `src/models/embedding_model.py` | GPU 推理（INT8）；单条/批量编码接口；输出 4096 维 Dense 向量（Sparse 由 Milvus BM25 承担）；单元测试（Mock）+ 集成测试 |
| B7 | BGE-Reranker-v2-minicpm-layerwise 客户端 | `src/models/reranker_model.py` | GPU 推理（与 Embedding 共享显卡）；layerwise 推理与 cutoff_layers 配置；输入 [query, doc] 对，输出相关性分数；超时回退机制；单元测试 |
| B8 | LLM 推理客户端（DashScope） | `src/models/llm_client.py` | 对接 DashScope OpenAI-compatible API；支持流式/非流式输出；单元测试（Mock） |

---

### 阶段 C：Ingestion Pipeline（MinerU → Chunk → Embedding → 存储）

**目的**：离线摄取链路跑通，能把样例 PDF 文档经 MinerU 解析 → 切分 → 增强 → 向量化 → 写入 Milvus + PostgreSQL（含 raw_documents 表存 MinerU 原始产物），支持幂等写入与增量更新。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| C1 | MinerU 产物加载器 | `src/rag/ingestion/mineru_loader.py` | 读取 `mineru_output/{name}/{backend}_auto/` 下的 `.md` + `content_list_v2.json`(v2 页级嵌套,见 2.4.4)；写入 PostgreSQL `raw_documents` 表；单元测试 |
| C2 | Chunking(父子分块 + 表格双粒度) | `src/rag/ingestion/chunking.py` + 单本书 anchor 配置 | **非表格内容**:按 §3.1.2 4 步流程切分 — Step 1 目录权威清单提取、Step 2 正文节边界匹配 + REAL_START 选取、Step 3 全书层面截断 + 节内参考文献丢弃、Step 4 父块构建(节本身或节内三遍切【】+(一)+1.,严格层级合并)、Step 5 子块构建(size 驱动,目标 600 字,父块 ≤ 1200 字直接当 child)。父块 `embedding_status='skip'` 不向量化,子块多向量化。**表格内容**(识别 MinerU `content_list` 中的 table 类型块):产出两类 chunk —— ① 整表摘要 chunk(LLM 一句概括) ② 逐行 chunk(parse HTML 转自然语言);共享 `parent_chunk_id` 指向所在节父块。**非表格图像类**:只留图名/bbox 到 `raw_documents.content_list`,不进 chunks 表(见 §3.1.1 末)。**单元测试**:目录字典提取 / REAL_START 选取规则 / 三遍切阈值 / 严格层级合并 / 子块 size 累积算法(含 force-add 边界) / 父子覆盖完整性 mismatch=0。**12 本书验证**:每本按 §3.1.2 切分主流程跑 + 抽样人审 + 验 mismatch=0 |
| C3 | 幂等性工具 | `src/rag/ingestion/idempotency.py` | source_id / heading_path_id / chunk_id / content_hash 生成逻辑与 3.1.4 一致；父块 chunk_id 固定使用 `relative_chunk_index="parent"` 参与哈希（见 3.1.4.2）；单元测试覆盖 normalize + 多级哈希 + 父块 ID 与子块 ID 不冲突验证 |
| C4 | LLM 语义增强 | `src/rag/ingestion/enrichment.py`、`src/agent/schemas/ingestion.py`（`ChunkEnrichmentOutput`，定义与 §9.5 一致） | 单次 LLM 调用生成 title / summary / hypothetical_questions（`tags` 字段已删除,见 §3.1.3.2 末尾决策）；Prompt 来自 `src/prompts/ingestion.py`；单元测试（Mock LLM） |
| C5 | 多向量 Embedding | `src/rag/ingestion/embedding.py` | 对每个 Chunk 生成 1 original + 1 summary + 2~3 question 向量记录；全部仅含 Dense 向量（Sparse 检索由 Milvus BM25 基于 original_content 文本字段实现）；批处理；content_hash 增量判断；单元测试 |
| C6 | 三层存储写入 + 僵尸清理 | `src/rag/ingestion/storage.py` | PostgreSQL chunks 表 upsert（父块 `embedding_status='skip'` 直写，跳过 Milvus）+ 子块 Milvus 向量 upsert + 僵尸 chunk 差集清理（先删子块含 Milvus，再删父块仅 PostgreSQL，见 3.1.4.3）；集成测试覆盖父块/子块删除顺序与 foreign key 约束 |
| C7 | Pipeline 编排 | `src/rag/ingestion/pipeline.py` | 串联 C1~C6；`python scripts/ingest.py <pdf_path>` 可完整摄取一份文档；集成测试 |
| C8 | 摄取入口脚本 | `scripts/ingest.py`、`scripts/init_db.py`、`scripts/init_milvus.py` | CLI 可用；支持单文件/批量摄取；初始化脚本可创建表结构和 Collection |

---

### 阶段 D:术语库(EL 已下线,数据备用)

**目的**:构建 `terms_collection`(2.4.6),导入 ICD-10-CN + CMeSH 标准术语数据。**EL 整层下线**后运行时不再被查询(原阶段 E 查询预处理术语扩展 / 阶段 F Agent 节点 ② build_query 的 Entity Linking / ④ extract_symptoms Tier 1/2 全部废弃),数据资产保留作为未来重新启用或切换到 EL_DESIGN_REVIEW §11.6 方案 A/B 时的基础。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| D1 | 术语数据整理与清洗 | `terms/icd10_cn/`、`terms/cmesh/` | 数据源格式统一为 `{concept_id, preferred_term, alias, source_vocab, icd10, category}` |
| D2 | 术语库构建脚本 | `terms/build_icd10.py` | 对 alias 文本做 Qwen3-Embedding-8B Dense 编码 → upsert 到 terms_collection；幂等；集成测试 |
| D3 | 术语检索接口 | `src/db/milvus/terms_collection.py` 扩展 | 输入口语文本，返回 Top-5 候选术语（含 concept_id / preferred_term / icd10）；按 category 可过滤；集成测试 |

---

### 阶段 E：Retrieval（Dense + Sparse + RRF + Rerank）

**目的**：在线查询链路跑通，输入用户 query，经预处理 → 双路召回 → 单阶段多路 RRF 融合 → 多向量聚合(sum-aggregate + vector_hits 副载荷) → 精排，输出 Top-K chunks（含 original_content + heading_path 引用信息），具备稳定回退策略。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| E1 | 查询预处理 | ~~`src/rag/retrieval/query_processing.py`~~ | **RETRIEVAL_EVAL §2 改造 + EL 移除后**:`sparse_queries` 改为 state 多字段直采(详见 §3.2.1 Step 2 / §4.1.2 ② Step 2),不再走 alias 反查 → `query_processing.py` 整文件删;Dense Query 整合改写 LLM 调用 + prompt 由 Agent ② `build_query` 节点持有(`src/prompts/agent.py` 的 `build_query_construction_prompt`) |
| E2 | Sparse Retriever（Milvus BM25） | `src/rag/retrieval/sparse_retriever.py` | 对 `sparse_queries` 中每个维度词袋分别查询 Milvus 内置 BM25，N 个维度 = N 次查询；各自返回 Top-N；单元测试 |
| E3 | Dense Retriever（单次 ANN） | `src/rag/retrieval/dense_retriever.py` | 对 `dense_query` 做 Qwen3-Embedding-8B 编码 → Milvus ANN 向量检索，返回 Top-N；单元测试 |
| E4 | 单阶段多路 RRF 融合 + 多向量聚合 | `src/rag/retrieval/fusion.py` | Dense（1 路）+ Sparse 各维度（各 1 路）→ 单阶段多路 RRF → 按 source_chunk_id 聚合(各 vector_type 命中分数求和 + 携带 `vector_hits` 副载荷) → Top-M;单元测试覆盖 sum-aggregate 公式与 vector_hits 形态(matched_text 三类取值规则) |
| E5 | Reranker 精排 + 回退 | `src/rag/retrieval/reranker.py` | Cross-Encoder 在 `diagnose` ⑩ 前置调用，对收敛后候选精排截断 Top-K；超时/不可用时回退至 `candidate_chunks` 原序（3.2.3 策略）；单元测试 |
| E6 | 元数据过滤（Pre-filter + Post-filter） | `src/rag/retrieval/` 各文件内 | Pre-filter：source_id 在 Milvus 检索时过滤；Post-filter：在 Rerank 前兜底过滤；missing → include 宽松策略 |

---

### 阶段 F：Agent 工作流（LangGraph StateGraph）

**目的**：按 4.1 节设计，使用 LangGraph StateGraph 实现完整诊断工作流（15 节点 + 2 条件路由）。先落地最小可用路径（① → ①.5 → ② → ③ → ④ → ⑩ → ⑪ → ⑫ → ⑬），再接入追问循环（⑤→⑥→⑦）和检查循环（⑧a→⑧b→⑨）。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| F1 | MedicalState 定义 + 初始化工厂 | `src/agent/state.py` | Pydantic `BaseModel`(见 §4.1.1 实现形态注)定义 34 字段;`present_illness_slots` 包含 12 个维度槽位(onset_time/onset_mode/trigger/location/nature/severity/duration_pattern/aggravating/relieving/associated_symptoms/progression/treatments),初始值为 None/空列表;实现 `create_initial_state(patient_id, patient_input) -> MedicalState` 工厂函数(初始值与 4.1.1a 节一致);单元测试。历史:已移除字段 `standardized_entities`(EL 下线)、`extracted_symptoms`(④ 节点废弃)、`info_gain`(信息增益机制移除);2026-05-22 PresentIllnessSlots schema 调整(trigger/nature/severity str→list[str];treatment_tried+response 合并为 treatments;13 维 → 12 维) |
| F2 | 节点 ①：info_collect | `src/agent/nodes/info_collect.py`、`src/agent/schemas/info_collect.py`（`InfoCollectOutput`，定义与 §9.5 一致） | Step 1: LLM 从 patient_input 提取 chief_complaint + present_illness + present_illness_slots（12 个维度槽位同步结构化填充，未提及维度保持 None/空）；Step 2: 以 patient_id 查 PostgreSQL 加载 medical_history；Step 3: 加载 exam_reports；Prompt 来自 `src/prompts/agent.py`；单元测试（Mock LLM + Mock DB）；验证：完整输入无空槽、简短输入多空槽 |
| F2.5 | 节点 ①.5：analyze_initial_reports | `src/agent/nodes/analyze_initial_reports.py`、`src/agent/utils/report_parser.py`、`src/agent/schemas/report_parser.py`(`ReportFinding` / `ReportFindings`,定义与 §9.5 一致) | exam_reports 非空时执行;多模态 LLM 直读报告(图片 jpg/png / PDF 直传)→ 提取 report_type / report_date / abnormal_values / impressions / positive_findings / negative_findings;输出 report_findings;exam_reports 为空时透传;Prompt 来自 `src/prompts/agent.py`;单元测试(Mock 多模态 LLM) |
| F3 | 节点 ②:build_query(EL 移除后整段重写) | `src/agent/nodes/build_query.py`、`src/agent/schemas/ner.py`(`NEREntity` / `NERResult`)、`src/agent/schemas/query_construction.py`(`QueryConstructionOutput`),Schema 定义均与 §9.5 一致 | 三步流程:Step 1 LLM NER 实体抽取(首轮对 `chief_complaint` + `present_illness`;后续轮仅对本轮新增 `followup_answer`);首轮把 symptom 类(temporality=current)按 negation 分流直接以 raw text 写 `confirmed_symptoms` / `denied_symptoms`(EL 删除后不再归一化)→ Step 2 `sparse_queries` 多字段直采(2026-05-17 RETRIEVAL_EVAL §2 改造:来源 A `chief_complaint` + 12 维 slots;来源 B `report_findings.positive_findings` 全加 + `impressions` 阴性过滤;去重 + 长度 ≥ 2)→ Step 3 `dense_query` 构建/改写(LLM 整合 confirmed + 已填维度 + 报告 positive/impressions 改写成一句自然语言);单元测试(Mock LLM) |
| F4 | 节点 ③：retrieve | `src/agent/nodes/retrieve.py` | 用改写后的 query 对 Milvus 做混合检索（Dense + Sparse 双路 → RRF 融合 → Top-N 截断），覆盖写入 `candidate_chunks`；单元测试 |
| F5 | 节点 ④:extract_symptoms(EL 移除后简化) | `src/agent/nodes/extract_symptoms.py` | 纯 TF-IDF 关键词提取,零 LLM,零 terms_collection 调用;输出每项 `{"text": kw, "preferred_term": None, "linked": False}`(后两字段保留兼容下游字段访问);单元测试 |
| F6 | 节点 ④:select_discriminative_symptom(1 LLM 重设计) | `src/agent/nodes/select_symptom.py`、`src/agent/schemas/symptom_selection.py`(`FollowupQuestion` / `UnaskableSymptom` / `SmartFollowupOutput`,定义与 §9.5 一致) | 1 次 LLM 调用同时出两件事:① `followup_questions`(slot 维度补全 + open 开放式追问,≤ MAX_FOLLOWUP_QUESTIONS,可为 0);② `unaskable_symptoms` 粗筛(LLM 想知道但患者答不上的体征,`{description, reason}`)。失败兜底:LLM 失败 → 返空 → 路由跳诊断。原"维度配额 + TF-IDF 信息增益 + 可问性评估"四步重工程化方案整体废弃 — 实测 TF-IDF 抽出 94% 教材通用高频词,信息增益的可比 key 立不起来。**所有阈值/上限常量来源见 §9.7,禁止 hardcode**;单元测试:slot+open 混合 / 信息已足返空 / unaskable 转储 |
| F7 | 条件路由：should_continue | `src/agent/routers/should_continue.py` | **纯函数路由**（不修改 State）：优先级：`followup_round >= settings.agent_limits.MAX_FOLLOWUP_ROUNDS`（常量来源 §9.7）→ 返回 diagnose（硬性兜底，防收敛失效无限循环；兜底 insufficient 产出由 Node ⑩ Step -1 完成）；`followup_questions` 非空 → followup；否则 → diagnose；所有自然收敛过滤逻辑（可问性、增益阈值）已内聚在 Node ④；单元测试覆盖三分支 + 验证路由函数调用前后 State 字段未被修改 |
| F8 | 节点 ⑤⑥⑦:追问循环 | `src/agent/nodes/generate_followup.py`、`wait_followup_answer.py`、`process_followup.py`、`src/agent/schemas/followup.py`（`FollowupParseResult`，定义与 §9.5 一致） | ⑤ LLM 将混合类型追问项（维度级 `type: "dimension"` + 症状级 `type: "symptom"`）转为患者可理解的流畅追问，写入 `followup_question`；⑥ `wait_followup_answer` 调用 interrupt() 等待用户回答（与 LLM 调用分离，避免恢复时重复生成）；⑦ LLM 解析回答：症状级 → 确认/否认/不确定三类分流更新 confirmed_symptoms / denied_symptoms / uncertain_symptoms；维度级 → 回填 `present_illness_slots` 对应槽位 + 追加 `present_illness` 自由文本；同时提取新增症状信息 → followup_round += 1 → 回到 build_query；Prompt 来自 `src/prompts/agent.py`；单元测试 |
| F9 | 节点 ⑧⑨：检查循环 | `src/agent/nodes/recommend_exam.py`、`wait_exam_report.py`、`process_exam_result.py` | ⑧a LLM 根据候选疾病推断所需检查（体格检查 + 辅助检查），按优先级排序；对与 report_findings 有交集的检查项，LLM 额外输出复用评估说明（含报告日期、采集条件判断），不静默删除，写入 `recommended_tests`；⑧b `wait_exam_report` 调用 interrupt() 等待结果回传（与 LLM 调用分离，避免恢复时重复生成）；⑨ 调用 report_parser.py 共享解析函数（与 ①.5 复用）→ 追加到 exam_reports 和 report_findings → 回到 build_query；`exam_round += 1`，上限 `settings.agent_limits.MAX_EXAM_ROUNDS`（常量来源 §9.7，禁止 hardcode）；单元测试 |
| F10 | 节点 ⑩：diagnose | `src/agent/nodes/diagnose.py`、`src/agent/schemas/diagnosis.py`（`RankedDisease` / `DiagnosisOutput`，定义与 §9.5 一致） | **Step -1 兜底短路**：入口直读 `state["followup_round"] >= settings.agent_limits.MAX_FOLLOWUP_ROUNDS`（常量来源 §9.7）时跳过 LLM，直接产出 insufficient 结果且 `failure_reason="followup_round_capped"`；正常路径：Step 0 Cross-Encoder 前置截断（可插拔，3.2.3）→ Step 0.5 父块扩展 + 同节图表 → **Step 1 1 步 LLM 诊断推理**(原生多模态 qwen3.5-plus,全量患者画像 + 文献 + figure 多模态喂入,直接出 `DiagnosisOutput` = results + retained_unaskable);**兜底 + 错误原因记录**:LLM 最多尝试 3 次仍失败 → try/except 捕获后返回 insufficient 结果,`failure_reason="step_1_structured_output_failed: <ExcType>: <msg>"`;同时 `logger.error(..., exc_info=True)` 记录完整堆栈;输出 diagnosis_result(含 disease / probability / evidence / differentiation / differentiation_type / failure_reason) + 覆盖 unaskable_symptoms(精筛供 ⑧a 消费);Prompt 来自 `src/prompts/agent.py:build_diagnose_prompt`(对齐 RAG 评测口径,3 步链已废弃);**单元测试覆盖 4 条路径**:① followup_round 触顶 → `failure_reason == "followup_round_capped"` ② 正常 1 步成功 → `failure_reason is None` + retained_unaskable 写回 ③ LLM 失败 → `failure_reason.startswith("step_1_structured_output_failed")` ④ vision LLM 路由验证(`settings.llm.VISION_*`);③ 断言 `differentiation_type == "insufficient"` 且 `probability == 0.0` 且 downstream 节点(⑪⑫⑬)仍能正常运行 |
| F11 | 条件路由：diagnose_router | `src/agent/routers/diagnose_router.py` | `need_exam` 且 `exam_round < settings.agent_limits.MAX_EXAM_ROUNDS` → recommend_exam；`confirmed` / `insufficient` / `exam_round >= settings.agent_limits.MAX_EXAM_ROUNDS` → safety_gate（常量来源 §9.7，禁止 hardcode）；单元测试 |
| F12 | 节点 ⑪：safety_gate | `src/agent/nodes/safety_gate.py`、`src/agent/schemas/safety_gate.py`（`SafetyGateOutput`，定义与 §9.5 一致） | 规则过滤：从 medical_history 提取过敏药物/当前用药/妊娠状态 → 匹配药物-过敏对（含同类药排除）+ 配伍禁忌表 + FDA 妊娠分级（D/X 禁用）；LLM 兜底：交叉过敏、罕见药物相互作用、肝肾功能剂量调整；输出 safety_constraints（banned_drugs / interaction_warnings / contraindication_flags）；单元测试 |
| F13 | 节点 ⑫⑬：建议与输出 | `src/agent/nodes/generate_advice.py`、`format_response.py`、`src/agent/schemas/advice.py`（`AdviceOutput`，定义与 §9.5 一致） | ⑫ 在 safety_constraints 约束内：confirmed → 用药建议 + 注意事项 + 复查建议；insufficient → 线下检查建议；need_exam 达上限 → 诚实告知局限；**读取 `diagnosis_result[0].failure_reason`**：`"followup_round_capped"` → risk_warnings 追加"问诊轮次较多仍未收敛"提示；`"step_N_structured_output_failed..."` → risk_warnings 追加"系统分析出现技术问题，结果不可作为依据"提示（不暴露异常细节）；高危提示优先级最高；⑬ LLM 组织自然语言回复 + 免责声明，failure_reason 非 None 时免责声明补一句"本次诊断因系统原因未能完整推理"；单元测试覆盖 failure_reason 的三种取值（None / followup_round_capped / step_N_... ）对 risk_warnings 和 final_response 的影响 |
| F14 | StateGraph 编排 | `src/agent/graph.py` | 注册 15 节点 + 2 条件边；顺序边 ①→①.5→②→③→④；条件边 ④→⑤/⑩（两路：追问或诊断）；追问循环 ⑤→⑥→⑦→②；检查循环 ⑧a→⑧b→⑨→②；诊断后路由 ⑩→⑧a/⑪；安全门控 ⑪→⑫；输出链 ⑫→⑬→END；集成测试 |
| F15 | 全工作流集成测试 | `tests/integration/test_agent_workflow.py` | Mock 存储 + Mock LLM，验证：正常路径（信息充足直接诊断）/ 追问循环（多轮追问后收敛）/ 检查循环（建议检查→结果回传→重新诊断）/ 信息不足路径 四条典型路径；安全门控过滤验证（过敏药物排除、配伍禁忌拦截） |

---

### 阶段 G：API 层与权限系统

**目的**：搭建 FastAPI 入口服务，实现 JWT 认证与角色权限（admin / patient），暴露问诊、患者信息、知识库管理等 RESTful 接口，接入限流保护。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| G1 | FastAPI 应用骨架 | `src/api/app.py`、`src/api/routes/__init__.py` | 应用可启动并挂载路由集合；注册 `prometheus-fastapi-instrumentator` 并暴露 `/metrics` 端点（**此时端点存在但无业务指标，业务埋点在 H2 完成，HTTP 层指标由 instrumentator 自动采集**）；**健康检查端点 `/healthz` + `/readyz` 在 H8 实现**，本任务不实现，不占用 `/health` 命名；启动冒烟测试 |
| G2 | JWT 认证中间件 | `src/api/middleware/auth_middleware.py`、`src/api/routes/auth.py` | 注册 / 登录 → JWT 签发；token 校验 + 角色提取；过期 / 无效 token 返回 401；单元测试 |
| G3 | 限流中间件 | `src/api/middleware/rate_limiter.py` | 基于内存的滑动窗口限流；超限返回 429；单元测试（进程内有效，多实例部署时由 H6 切换为 Redis 后端以共享配额） |
| G4 | 问诊接口 | `src/api/routes/diagnosis.py`、`src/api/schemas/diagnosis_schema.py` | `POST /diagnose`：调用 Agent graph → 返回诊断结果；支持追问交互（session_id 关联）；**按 §9.6 规则在返回响应前写入一条 `rag_trace` 记录**（从 final State 组装 15 字段，含 `last_reranked_chunks` / `last_diagnose_prompt` / `last_diagnose_raw_output` / `session_token_usage` / `session_latency_ms`）；集成测试（含"`POST /diagnose` 完成后对应会话在 `rag_trace` 表有一条完整记录"断言） |
| G5 | 患者信息接口 | `src/api/routes/patient.py`、`src/api/schemas/patient_schema.py` | 患者 CRUD（仅 patient 角色可操作自己的数据）；集成测试 |
| G6 | 管理员接口 | `src/api/routes/admin.py` | 知识库上传（触发 Ingestion Pipeline）/ 系统配置修改 / 用户管理；仅 admin 角色；集成测试 |
| G7 | Nginx 反向代理 | `infra/docker/nginx.conf`、`docker-compose.yml` 更新 | Nginx 代理 FastAPI；HTTPS（可选）；健康检查 |

---

### 阶段 H：基础设施增强（监控、缓存、日志）

**目的**：接入 Prometheus + Grafana 指标监控，Promtail + Loki 日志采集，落地 Redis 缓存客户端并与业务层对接，完善 config/logging_config.py。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| H1 | Redis 缓存客户端 | `src/db/redis/cache.py` | **仅实现配置缓存**（`config:<key_name>`，TTL 60s）读写测试通过；**不实现 RAG 响应级缓存**（原因见 §5.1"为何不做 RAG 响应级缓存"——Agentic 工作流下同一 query 在不同 State 应产出不同结果，query 字符串做 key 会串话）；Redis 不可用时降级（回源 PG）测试通过 |
| H2 | Prometheus 指标埋点 | `src/common/metrics.py` 完善、`infra/prometheus/prometheus.yml` | **基础指标**：向量检索耗时 / LLM 调用耗时 / Token 统计 / QPS / 错误率；**上下文/会话级指标**（对应 4.2.7 上下文表）：`context_tokens_per_llm_call` / `context_structured_fields_size` / `context_messages_count` / `context_loop_iterations`；**结构化输出健康度**（对应 4.2.7 结构化输出表，**调 prompt/Schema 的核心可观测面**）：`structured_output_attempt_total` / `retry_total` / `failure_total` / `fallback_triggered_total` / `diagnose_failure_reason_total` / `latency_seconds`，按 `node` + `schema` 标签分桶（`failure_total` 额外按 `exception_type`、`fallback_triggered_total` 额外按 `fallback_type`、`diagnose_failure_reason_total` 按 `reason_kind`）；**实现方式（明确约束）**：① `src/common/metrics.py` 仅做**模块级指标对象声明**（6 个指标作为单例，import 使用）+ 定义 `RetryObserver(BaseCallbackHandler)` 用于捕获 `with_retry` 内部重试事件；② **禁止**封装装饰器 / helper 函数 / 上下文管理器；③ 各 LLM 调用点按 §9.1 伪代码模板**裸写** `try/except/finally` 手动 `.inc()` / `.observe()`，调用时通过 `config={"callbacks": [retry_observer], "metadata": {...}}` 传入 RetryObserver；Prometheus 可抓取；**本任务只负责 Prometheus 聚合时序指标，与 G4 的 `rag_trace` per-session DB 记录是两个独立系统（见 §9.6），数据不重复**；**MVP 基线（本任务内完成）**：④ 接入 `prometheus-fastapi-instrumentator` 暴露 HTTP 层指标（`http_request_duration_seconds` 等，§5.2.1 ②），一行代码注册；⑤ 依赖层指标（§5.2.1 ③）——SQLAlchemy 事件订阅（`src/db/postgres/metrics.py`）、Redis Histogram wrapper（≤ 20 行）、Milvus client wrapper（`src/db/milvus/client.py`）；⑥ `/metrics` 端点排除 `/healthz`/`/readyz`/`/metrics` 自身避免自污染；⑦ **演进路径指标不在本任务范围**（见 §5.2.5），严禁提前埋点；单元测试：对一个典型调用点断言成功/重试/失败/兜底四种路径下对应指标的增量正确；对 `RetryObserver` 单独断言 `on_retry` 触发时 `_retries` 正确累加；HTTP/依赖指标通过 `GET /metrics` 断言能看到对应 metric family |
| H3 | Grafana 仪表盘 | `infra/grafana/dashboards/` | 应用性能仪表盘 + 硬件资源仪表盘；导入即可用 |
| H4 | 日志采集（Promtail → Loki） | `config/logging_config.py`、`infra/promtail/promtail-config.yml`、`infra/loki/loki-config.yml` | 诊断日志 / 错误日志 / 访问日志 写入 Loki（**JSON 结构化格式**，强制字段见 §5.2.1.1：`trace_id` / `session_id` / `patient_id` / `node` / `level` / `message` / `exc_info` / `timestamp`）；`config/logging_config.py` 用 `python-json-logger` 的 `JsonFormatter` + `contextvars.ContextVar` 在 FastAPI middleware 入口注入 `trace_id`，与 G4 写 `rag_trace` 用同一个 UUID 打通审计↔日志；审计日志写入 PostgreSQL；Grafana 可通过 `trace_id` label 在 Loki 中检索并关联到 PG `rag_trace` 详情 |
| H5 | Node Exporter 硬件监控 | `docker-compose.yml` 更新 | CPU / 内存 / 磁盘 / 网络指标采集；Grafana 仪表盘可视化 |
| H5b | DCGM Exporter GPU 监控 | `docker-compose.yml` 更新 | GPU 使用率 / 显存 / 温度 / 功耗指标采集（NVIDIA DCGM Exporter :9400）；Grafana GPU 仪表盘 |
| H6 | Redis 缓存与业务层对接 | `src/db/redis/cache.py` 与业务层对接、`src/api/middleware/rate_limiter.py` 改造 | 动态配置缓存（60s TTL）生效；冷启动首次读取回源 PG 并写缓存；**将 G3 的内存滑动窗口限流替换为 Redis 后端**（多实例共享配额，SCRIPT `INCR` + `EXPIRE` 原子操作）；集成测试（**不含 RAG 响应缓存**，见 H1） |
| H7 | 动态配置管理 | `src/db/postgres/` 中 system_config 表 | system_config 表存储 Top-K / 温度 / 阈值等；服务定时读取（经 Redis 缓存）；admin 可通过 API 修改 |
| H8 | 健康检查端点 | `src/api/routes/health.py` | 实现 `GET /healthz`（liveness，零依赖，固定 200）+ `GET /readyz`（readiness，并发探测 PG `SELECT 1` + Milvus ping，2s 超时，任一失败返 503 且响应体列出 `failing`；Redis 不可用仍算 ready，对齐 §5.1 降级模式）；两个端点不经过 JWT/限流/审计；详见 §5.2.4；单元测试覆盖所有依赖正常 / PG 失败 / Milvus 失败 / 同时失败 4 条路径 |

---

### 阶段 I：评估体系

**目的**：实现离线评估（RAG 检索质量 + Agent 决策质量）、在线追踪（端到端延时 + Token 统计）、LLM Judge 评分，建立回归基线。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| I1 | RAG 离线评估 | `evaluation/offline/rag_evaluator.py`、`evaluation/datasets/rag_eval.jsonl`、`src/agent/schemas/evaluation.py`（首次创建，含 `FaithfulnessScore` / `ClaimJudgment` / `RelevanceScore` / `HallucinationReport`，定义与 §9.5 一致；I2/I3 继续追加 Schema 到本文件） | 召回率 / 准确率 / MRR 指标计算；Golden Test Set 构建；可复现执行 |
| I2 | Agent 离线评估（L1~L5 梯度） | `evaluation/offline/agent_evaluator.py`、`evaluation/datasets/agent_eval.jsonl`、追加到 `src/agent/schemas/evaluation.py`（`DecisionTraceScore`，定义与 §9.5 一致） | Mock 检索结果 → 评估 Agent 决策链；覆盖 L1（完整信息）~ L5（矛盾信息）五个梯度；工具选择准确率 / 自我纠错能力 / 幻觉决策 等维度 |
| I3 | LLM Judge | `evaluation/offline/llm_judge.py`、追加到 `src/agent/schemas/evaluation.py`（`ResponseQualityScore` / `AdviceCompletenessScore`，定义与 §9.5 一致） | Prompt 来自 `src/prompts/evaluation.py`；评估响应质量 + 轨迹合理性；输出评测报告 |
| I4 | 在线追踪 | `evaluation/online/tracing.py` | 端到端延时 / 每步 Token 统计 / 每次运行上报；阈值告警 |
| I5 | 评估脚本入口 | `scripts/` 或 `evaluation/` 中的 runner | `python -m evaluation.offline.rag_evaluator` / `agent_evaluator` 一键执行 |

---

### 阶段 J：端到端验收与文档收口

**目的**：在真实环境（非 Mock）下跑通全链路 E2E 冒烟测试——与前面各阶段使用 Mock 存储 / Mock LLM 的集成测试不同，本阶段使用真实运行的 PostgreSQL、Milvus、LLM 推理后端，验证数据真正在各层之间流转。完善 README，确保"开箱即用 + 可复现"。

| 编号 | 任务 | 产出文件 | 验收标准 |
|------|------|---------|---------|
| J0 | Docker 化部署 | `infra/docker/Dockerfile.api`、`.dockerignore`、`docker-compose.yml`（如需） | api 容器可成功 `docker compose build api` 通过；`docker compose up -d api nginx` 全栈起来 + healthcheck 转绿；`/healthz` 经 nginx → api 链路通；H8 readyz 探到容器内 PG/Milvus；H2 / H4 业务指标与日志通过容器化的 api 流到 Prometheus / Loki（应用性能 dashboard 不再空） |
| J1 | E2E：Ingestion 全链路 | `tests/e2e/test_ingestion_e2e.py`（与 C7 的 `tests/integration/test_ingestion_pipeline.py` 区分：C7 是 Mock 依赖的集成测试，J1 是真实 Milvus/PG 的端到端冒烟） | 样例 PDF → MinerU 产物 → 完整摄取 → Milvus + PostgreSQL（含 raw_documents 表）数据验证 |
| J2 | E2E：Retrieval 全链路 | `tests/e2e/test_retrieval_e2e.py`（走真实 Milvus + 真实 Embedding/Reranker） | 真实 query → 双路召回 → RRF → Rerank → Top-K 结果校验 |
| J3 | E2E：Agent 全链路 | `tests/e2e/test_agent_workflow_e2e.py`（**与 F15 的 `tests/integration/test_agent_workflow.py` 区分**：F15 走 Mock LLM + Mock DB 的集成测试，J3 走真实 DashScope + 真实 Milvus/PG 的端到端冒烟） | 模拟患者输入 → 完整工作流 → 诊断输出（覆盖正常/高危/追问/低置信度路径） |
| J4 | E2E：API 接口 | `tests/e2e/test_api_e2e.py`（真实 FastAPI + 真实后端） | 注册 → 登录 → 问诊 → 追问 → 获取结果 完整交互链路 |
| J5 | README 完善 | `README.md` | 项目介绍 / 快速开始 / 环境要求 / Docker 部署 / 配置说明 / API 文档 / 评估运行 |
| J6 | 清理与一致性检查 | 全项目 | 无未使用的 import / 无空实现桩 / 类型注解完整 / 全部测试通过 |

---

## 8.4 进度跟踪表

> 状态说明：`[ ]` 未开始 | `[~]` 进行中 | `[x]` 已完成
>
> 更新时间：每完成一个子任务后更新对应状态

### 阶段 A：工程骨架与基础设施基座

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| A1 | 初始化目录树与最小可运行入口 | [x] | 2026-05-01 | §1.3.1 目录树修订(删过时 `data/` + 简化 `terms/`);`src/__init__.py` + `src/__main__.py` 就绪,`python -m src` 通过 |
| A2 | Docker Compose 搭建存储基座 | [x] | 2026-05-12 | compose 配齐 + Milvus 三件套 + PG + Redis 5 服务全部 healthy(`docker compose ps` 验证),端口绑 127.0.0.1 防外网。监控类服务(Prometheus/Grafana/Loki/Node Exporter/DCGM)留 H 阶段验收 |
| A3 | 配置加载与校验 | [x] | 2026-05-01 | `config/settings.py` 实现(§9.7 + `LLM_API_KEY` 必填),5 测试 PASS;spec 同步删除冗余的 `config/model_config.py` |
| A4 | pytest 测试基座 | [x] | 2026-05-12 | tests/{unit,integration,e2e} 三层目录就绪,257 PASS / 3 skip(GPU 模型 skip);CLAUDE.md 加测试位置规则;conftest YAGNI(无共享 fixture 需求)。`test_reranker_smoke` / `test_terms_retrieval_smoke` 由本地 GPU 任务跑过 |
| A5 | 公共工具模块 | [x] | 2026-05-12 | `src/common/normalize.py`(C3 用)+ `hashing.py`(SHA256,§3.1.4 各 ID 派生)+ `metrics.py`(F 阶段补,§9.1 6 指标 Prometheus 单例 + RetryObserver);单测覆盖 normalize/hashing,metrics 在节点 unit 测里间接验证 |
| A6 | Prompt 模板骨架 | [~] | | 已:`src/prompts/ingestion.py`(C4 enrichment + figure summary)+ `agent.py`(F 阶段 14 prompt builders);待:`src/prompts/evaluation.py`(I 阶段 LLM Judge,各 evaluator 写时再补) |

### 阶段 B：数据层与模型客户端

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| B1 | PostgreSQL 连接池 + ORM 模型 | [x] | 2026-05-12 | `connection.py` + `models.py`(Source/RawDocument/Chunk)+ `models_patient.py`(User/Patient + 8 张患者历史表 §2.4.5)+ `models_dialog.py`(Session/Conversation §2.4.3)+ `models_audit.py`(RagTrace 15 字段/KbChangeLog/ConfigChangeLog/DiagnosisFeedback/SystemConfig §5.2.3 + §5.3),共 20 张表统一挂在 `Base.metadata`;`__init__.py` 集中导出 + Alembic env.py 自动扫描;26 unit(15 老 + 20 新)+ 24 integration(11 老 + 12 新 patient/dialog/audit CRUD + 1 alembic upgrade head)PASS |
| B2 | PostgreSQL 迁移脚本 | [x] | 2026-05-12 | 6 个 SQL 文件齐(0001-0003 老 + 0004 users_patients + 0005 sessions_conversations + 0006 audit_config),含 patient cascade / sessions partial index `WHERE status='active'` / medications partial index `WHERE ended_at IS NULL` / rag_trace 3 索引等;Alembic 接入(`alembic.ini` + `env.py` 走 `settings.postgres.dsn` + `_helpers.execute_sql_file` 用 `exec_driver_sql` 绕开 `:name` bind 解析)+ baseline 现有 PG;`tests/integration/test_alembic_upgrade.py` 在隔离 schema 跑通 `alembic upgrade head` 验全部 20 张表 + 关键索引 + alembic_version=head |
| B3 | Milvus 连接管理 + docs_collection | [x] | 2026-05-01 | `config/milvus_schema.py` 9 字段 schema(spec 8 + BM25 派生 sparse)+ HNSW dense / BM25 sparse(中文 analyzer)/ 3 scalar 索引;`src/db/milvus/docs_collection.py` ensure/upsert/search_dense/search_sparse_bm25/count/drop;9 unit + 5 integration PASS(中文 BM25 命中"胆囊炎"验证) |
| B4 | Milvus terms_collection | [x] | 2026-04-30 | schema(8 字段)+ HNSW + INVERTED 索引 + ensure/upsert/search/count/drop 接口齐;`config/milvus_schema.py` + `src/db/milvus/terms_collection.py` |
| B5 | PostgreSQL raw_documents 表（MinerU 产物存储） | [x] | 2026-05-02 | `connection.py`(engine + session_scope)+ `models.py`(Source/RawDocument ORM + `upsert_source`/`upsert_raw_document` 幂等接口,ON CONFLICT DO UPDATE);4 个 JSONB 字段全 NOT NULL(spec §2.4.4 修订版);11 测试 PASS(6 unit schema 锁 + 5 integration:upsert/幂等/级联删/GIN jsonpath 查 type/FK 违反) |
| B6 | Qwen3-Embedding-8B 客户端 | [x] | 2026-05-11 | `src/models/embedding_model.py`:`EmbeddingModel`(8bit BitsAndBytesConfig + device_map='auto' + SentenceTransformer)+ encode/encode_one + `get_embedding_model()` 进程内 singleton(双检锁)。`DEFAULT_BATCH=8`(实测 batch=16 在长文本 chunk 上偶发 CUDA OOM,降到 8 后稳态显存 ~12GB 杜绝 OOM,吞吐量未损)。`terms/build_icd10.py` 已复用该 singleton 删本地 `load_embedding_model()` |
| B7 | BGE-Reranker-v2-minicpm-layerwise 客户端 | [~] | | 已:`src/rag/retrieval/reranker.py` 封装 LayerWiseFlagLLMReranker(lazy load + `cutoff_layers` 走 §9.7 `agent_limits.RERANKER_CUTOFF_LAYERS`)+ `rerank_with_fallback` 高阶函数(timeout / fallback,9 unit PASS,在 F10 diagnose Step 0 已被使用);**还差**:`tests/integration/test_reranker_smoke.py` 真 GPU 跑过(模型权重已下载到 RERANKER_MODEL_PATH,需 GPU 空闲时跑一次锁住 layerwise 切层与排序行为) |
| B8 | Qwen LLM 推理客户端 | [x] | 2026-05-01 | `src/models/llm_client.py::get_llm()` 工厂(lru_cache,薄层不封装 retry/metrics 按 §9.1);7 测试 PASS(5 unit + 2 integration smoke 真调 DashScope qwen3.5-122b-a10b 通,流式 + 非流式) |

### 阶段 C：Ingestion Pipeline

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| C1 | MinerU 产物加载器 | [x] | 2026-05-02 | `src/rag/ingestion/mineru_loader.py::load_mineru_output()`(177 行)读 4 文件 + **双清洗 image VLM 幻觉**(v2 `block.content.content` 删除 + 用同段文本作指纹精确 substring 删 markdown,短指纹 < 20 字符跳过防误删,清洗后 grep 自检 unclean 报 warning)+ source_id 走 C3 + upsert sources/raw_documents + 返回 stats dict(预留 H2/§5.2.3 埋点接口);保留 image_caption / image_footnote / bbox / `![](images/...)` 占位符 / table.html / chart.content / page_header 等(过滤归 C2);11 unit + 3 integration PASS;`scripts/load_mineru.py` 批量入口(单本/--all);**13 本教材全部灌入 PG**(13912 页 / 264948 block / 删 7426 image content / 0 指纹遗漏 / 22.6s,raw_documents 表占 273MB);顺手删 0 行僵尸文件 `image_caption.py` |
| C2 | Chunking(父子分块 + 图表 chunk) | [~] | 2026-05-12 | **POC 12 本切分完成,全部 mismatch=0**(~12k 父块 + ~25k 子块);通用 SOP 见 [scripts/METHODOLOGY.md](scripts/METHODOLOGY.md) + 各书 BOOK_NOTES。**12 本图表 manifest 流水线就绪**:`extract_figures.py` 出 4026 条 manifest;`derive_figure_heading_paths.py` 按 (pg_start, head 前缀) 反查 POC parent 关联 heading_path(3891 hit / 135 孤儿);`merge_multipanel_figures.py` 多面板合并 27 anchor 吸 33 sibling;`merge_crosspage_tables.py` 跨页冗余 89 anchor 吸 91 dup;`reroute_figure_in_table.py` 16 条 chunk_kind 误标修正。**chunks 表落库就绪**:`load_chunks_to_pg.py` 灌 31063 text 行 + `load_media_chunks_to_pg.py` 灌 3767 图表行(单行多列设计,2026-05-12 重构自原"源+summary"双行,见 migration 0003 / spec §3.1.2)。**block extractor `extract_chunkable_text` 已实现**(15 unit PASS)。**还差**:Step3 POC port 到 production `src/rag/ingestion/chunking.py` 主流程(MVP 阶段不阻塞 — 12 本书已切完入库,新书入库时再做) |
| C2.5 | 用药指南专用处理(待定) | [ ] | | **用户已确认 deferred**(2026-05-12),用药指南 30289 条药品后续录入,F12 safety_gate 规则层暂用结构化 `medical_history` 直接抽取过敏/妊娠 + LLM 兜底。背景与候选方案见 spec §8.3 C2.5 / pending tasks。**还差**:整个任务,待用药指南规则表录入(用户主导) |
| C3 | 幂等性工具 | [x] | 2026-05-02 | `src/rag/ingestion/idempotency.py` 6 个纯函数(normalize / source_id / heading_path_id / chunk_id / parent_chunk_id / content_hash);全部按 §3.1.4 规则,无 IO 无状态;30 unit PASS(覆盖 normalize 6 个、source_id 5、heading_path 5、chunk_id 3、parent 3、content_hash 4 + 综合 4) |
| C4 | LLM 语义增强 | [x] | 2026-05-12 | **三路 enrichment 全跑完且入 PG**:① child 22287/22287(deepseek-v4-pro,`scripts/enrichment.py`)→ `load_enrichment_to_pg.py` 灌 title/summary/hq 3 字段;② figure(原 chart + flowchart)1023/1023(qwen3.5-plus 多模态 vision,`figure_enrichment_generation.py`,0 fail);③ table 2744/2744(deepseek-v4-pro,`table_enrichment_generation.py`,0 fail)→ `load_media_chunks_to_pg.py` 单步 4 字段灌入(含 medical_statement)。**单步 4 字段设计**(替代原 spec "两步":vision 看图直出 4 字段,避 Stage 2 enrichment 看不到图的视觉幻觉扩散);schema `FigureSummaryEnrichmentOutput`(§9.5)+ shared prompt tail `_SHARED_4FIELD_TAIL`(临床/口语 question 分布、英中混排、caption 杂质识别)+ footnote 必传(强调解读缩写/图例,不写参考文献入 ms);deepseek 走 `method="json_mode"` 避 json_schema BadRequest。**2026-05-12 单行多列重构**:删 tags / linked_chunk_id / bm25_only;content_hash 改 B 方案 `SHA256(chunk_raw_text + medical_statement)` 让 mineru 或 LLM 任一变化都触发重 embed |
| C5 | 多向量 Embedding | [x] | 2026-05-11 | `src/rag/ingestion/embedding.py::build_milvus_records` 把 PG chunks 行扩展为 1 original + 1 summary + N question 条 Milvus 记录(child 用 chunk_raw_text 作 dense,table/figure 用 medical_statement;text_for_bm25 仅 original 非空,符合 §3.1.5.1)。`scripts/load_chunk_embeddings_to_milvus.py` 按 source_id 分批 → encode → upsert → PG 标 done(异常 → 标 failed 可重置重跑,幂等)。**全量灌入完成**:26054 chunks → **Milvus 129810 entities**(平均 4.98 records/chunk),0 failed / 0 pending;首次 batch=16 全量耗时 58 min 含 3 次 OOM 共 900 chunks 标 failed,改 batch=8 后 2.2 min 补跑完毕 0 fail。端到端检索 sanity 通过(`急性胸痛伴出汗如何处理` → Dense 命中 question 向量、BM25 命中 original) |
| C6 | 三层存储写入 + 僵尸清理 | [ ] | | **写入部分实际已完成**(scripts/load_chunks_to_pg.py + load_media_chunks_to_pg.py + load_chunk_embeddings_to_milvus.py 走 ON CONFLICT DO UPDATE 幂等灌入,12 本书 26054 chunks → 129810 Milvus entities 全成功)。**还差**:① `src/rag/ingestion/storage.py` 把散落 scripts 收口到 production 模块;② 僵尸 chunk 差集清理(重灌时 PG 旧 chunk 删除 + 对应 Milvus 子向量先删 — spec §3.1.4.3)。**MVP 阶段不阻塞**(不重灌就不需要清理) |
| C7 | Pipeline 编排 | [ ] | | scripts 里散落 `load_mineru.py` / `enrichment.py` / `figure_enrichment_generation.py` / `table_enrichment_generation.py` / `load_*_to_pg.py` / `load_chunk_embeddings_to_milvus.py` 构成事实上的 pipeline。**还差**:`src/rag/ingestion/pipeline.py` 把这些串成一个统一函数。**MVP 阶段不阻塞**(scripts 已能跑通新书入库) |
| C8 | 摄取入口脚本 | [ ] | | scripts 里有 `batch_parse_pdfs.sh`(MinerU 批量解析)+ 各 `load_*.py` 灌库脚本,事实上能跑批。**还差**:① `scripts/ingest.py` 单文件 CLI 入口;② `scripts/init_db.py` PG 表创建脚本(目前手工跑 `migrations/0001_*.sql` / `0002_*.sql`);③ `scripts/init_milvus.py` Milvus collection 创建脚本(目前 `ensure_*_collection` 在首次调用自动建表)。**MVP 阶段不阻塞**(C7/C8 一起做,等 production pipeline 收口时一并补) |

### 阶段 D:术语库(EL 已下线,数据备用)

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| D1 | 术语数据整理与清洗 | [~] | | 已:ICD-10 北京临床版 v601 下载就位(/data/medical-resources/ICD10/,40k 行)+ `terms/build_icd10.py` 内嵌清洗(去重、空值过滤、类型推导)。**还差(YAGNI 暂不做)**:CMeSH 等其他术语源 — 当前每 ICD 编码只灌 1 别名(=preferred_term 自身),E1 Sparse 词袋实际单 token,等口语词表灌库后多别名效果才完整。用户已确认暂不做 |
| D2 | 术语库构建脚本 | [x] | 2026-04-30 | `terms/build_icd10.py` 灌入 40474 条 ICD-10 北京临床版;主键 `{icd_code}_{SHA256(alias)[:16]}` + Milvus upsert 保证幂等;categorize:R 段→symptom,其他→disease |
| D3 | 术语检索接口 | [x] | 2026-04-30 | `search_aliases` 候选池放大 + 按 `preferred_term` 去重(score 同则取 concept_id 更短/字母序更小);确定性 tie-break 保证幂等;冒烟测试 11 条 query Top-1 命中率 95%、Top-K 信息密度 5/5 |

### 阶段 E：Retrieval

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| E1 | 查询预处理(RETRIEVAL_EVAL §2 改造 + EL 移除后 sparse 路重构) | [x] | 2026-05-18 | **当前生产实现**:`sparse_queries` 由 `src/agent/nodes/build_query.py` Step 2 state 多字段直采(chief + 12 维 slots + report findings,详见 §3.2.1 Step 2 / §4.1.2 ②);**已删**:`src/rag/retrieval/query_processing.py`(原 `expand_aliases` / `build_sparse_query_bag` / `build_sparse_queries` 三函数)+ 配套 unit/integration 测试。Dense Query LLM 改写由 F3 持有 |
| E2 | Sparse Retriever（Milvus BM25） | [x] | 2026-05-12 | `src/rag/retrieval/sparse_retriever.py::search_sparse_routes` 高阶函数:循环调底层 `docs_collection.search_sparse_bm25`,N 个维度 = N 次 BM25(顺序保留);默认 top_k = `settings.agent_limits.RETRIEVE_TOP_N`(spec gap:E2 说"返回 Top-N"未明示数字,按 §9.7 取齐);`source_id_filter` pre-filter 透传(对接 E6);6 unit + 4 integration PASS(临时 docs_collection 隔离,验证多维度命中、跨维度共命中、source pre-filter、空入入兜底) |
| E3 | Dense Retriever（单次 ANN） | [x] | 2026-05-12 | `src/rag/retrieval/dense_retriever.py::search_dense_route(dense_query, top_k=None, source_id_filter=None)`:文本经 `get_embedding_model().encode_one` → `docs_collection.search_dense`,不传 vector_type_filter(spec §3.2.2 三类向量均参与召回);默认 top_k = `settings.agent_limits.RETRIEVE_TOP_N`;6 unit + 3 integration PASS(真 Embedding 8B INT8 + 真 Milvus 临时 collection,3 条临床主题 chunk 验证语义命中:右上腹剧痛 query 命中胆囊炎 chunk Top-1) |
| E4 | 单阶段多路 RRF 融合 + 多向量聚合 | [x] | 2026-05-12 | `src/rag/retrieval/fusion.py::fuse_routes(dense, sparse_routes, top_n, rrf_k=60, pg_chunk_lookup)`:5 步流程(record-level RRF 求和 → chunk-level 多向量聚合 → top_n 截断 → PG 仅对存活 chunk 回查 summary/hypothetical_questions → 装配 vector_hits);matched_text 三类规则按 spec §3.2.2 行 1822-1825(original 直读 hit / summary+question 走 PG lookup);question vector ID 解析 `_q{n}` 后缀(spec §3.1.6.2 + §3.2.2 已与代码对齐统一为 `_q{n}`);PG lookup 注入设计便于单测 mock,只对截断存活且需 summary/question 的 chunk 调用节省 IO;21 unit PASS(覆盖 RRF 公式/跨路求和/多向量聚合/top_n 截断/同分字母序 tie-break/vector_hits 形态/matched_text 三类/lookup 调用优化/边界) |
| E5 | Reranker 精排 + 回退（diagnose ⑩ 前置） | [x] | 2026-05-12 | `src/rag/retrieval/reranker.py::rerank_with_fallback(query, documents, top_k, timeout_sec=10, enabled=True, reranker=None)` 高阶函数:用 ThreadPoolExecutor 实现 best-effort 超时;**spec §3.2.3 强约束 - 不抛异常**(模型异常 / 超时 / disabled 三种 fallback 路径都返回原序 idx 列表);返回 `list[int]` 解耦 candidate_chunks 形态;9 unit PASS(正常排序/top_k 截断/disabled 跳过/空入入/异常 fallback/超时 fallback/timeout=None 不限时)。底层 `Reranker` / `get_reranker` 沿用 B7 实现 |
| E6 | 元数据过滤 | [x] | 2026-05-12 | Pre-filter 已在 E2/E3 通过 `source_id_filter` 透传到底层 Milvus(覆盖 spec §3.2.3 "硬约束在底层索引提前过滤"),本任务补 Post-filter 框架:`src/rag/retrieval/metadata_filter.py::apply_post_filters(candidates, predicates)` 接受 predicate list(签名 `dict → bool|None`),全部 True/None 才保留;**spec §3.2.3 'missing → include' 宽松策略**实现:返回 None / 抛 KeyError/AttributeError/TypeError 视为通过,其他异常冒泡;`source_id_in_allowlist` factory 演示常用 predicate 模式;14 unit PASS(空 predicates / AND 语义 / 三类 missing 异常宽松 / 其他异常冒泡 / 顺序保留 / factory 行为) |

### 阶段 F：Agent 工作流

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| F1 | MedicalState 定义 + 初始化工厂 | [x] | 2026-05-01 | `src/agent/state.py` 实现(§4.1.1 37 字段 **Pydantic BaseModel** + 嵌套 `PresentIllnessSlots` / `SessionTokenUsage` / `SessionLatencyMs` + `create_initial_state`),8 测试 PASS(字段清单 + 类型校验 + 老数据反序列化默认值 + 多 session 不共享 + §9.2 演化规则) |
| F2 | 节点 ①：info_collect | [x] | 2026-05-12 | LLM Step1 + DB Step2/3(占位见 `utils/patient_repo.py`，B 阶段建表后接真实 ORM）；3 unit PASS |
| F2.5 | 节点 ①.5：analyze_initial_reports | [x] | 2026-05-23 | 多模态报告解析共享逻辑落 `utils/report_parser.py`(⑨ 复用,带 hint);**2026-05-23 X4 真传后端**:resume value 改 `list[group dict]` 跟 ⑧b/⑨ 共享 `exam_results` 字段,上传路径走 parse_reports + group_label hint,跳过路径走 PG fallback;前端分框 UI(N 框 = N 报告,框内 multiple,标签选填);6 unit PASS(覆盖 skip/PG fallback/LLM 失败/单组/多组含 skipped/无 label) |
| F3 | 节点 ②：build_query | [x] | 2026-05-12 | 4 步：NER → EL（每实体一次）→ 确定性术语扩展 → Query 构建；首轮自动初始化 confirmed/denied；check path 跳 Step1/2；3 unit PASS |
| F4 | 节点 ③：retrieve | [x] | 2026-05-12 | dense + sparse 双路 + RRF 融合；PG callable 注入（`utils/chunks_lookup.py`）补 summary/question matched_text；3 unit PASS |
| F5 | 节点 ④：extract_symptoms | [x] | 2026-05-12 | 零 LLM：TF-IDF 关键词 + 三层归一化（Tier1 alias 精确 / Tier2 向量阈值 / Tier3 保留原文）；3 unit PASS |
| F6 | 节点 ④：select_discriminative_symptom | [x] | 2026-05-12 | 维度缺口配额制（≤2）+ 报告证据消费 + 已问过滤（Tier3 软比对）+ 信息增益贪心 + 可问性评估循环 + 阈值兜底；4 unit PASS |
| F7 | 条件路由：should_continue | [x] | 2026-05-12 | 纯函数，三分支（cap / questions / 其他）；4 unit PASS（含不修改 State 验证） |
| F8 | 节点 ⑤⑥⑦:追问循环 | [x] | 2026-05-12 | ⑤ 自由文本生成 + ⑥ interrupt 等待 + ⑦ structured 解析（三类回答 + 维度回填 + present_illness 追加）；5 unit PASS |
| F9 | 节点 ⑧⑨：检查循环 | [x] | 2026-05-12 | ⑧a 自由文本检查推荐 + ⑧b interrupt 等待 + ⑨ 复用 report_parser（全局 report_index 重映射）；4 unit PASS |
| F10 | 节点 ⑩：diagnose | [x] | 2026-05-19 | Step-1 cap 短路 + Step0 reranker（已有 fallback）+ Step0.5 父块扩展(删 vector_hints) + **1 步 LLM 诊断**(原生多模态 qwen3.5-plus,对齐 RAG 评测口径) + LLM 失败兜底 insufficient + failure_reason + last_diagnose_prompt/raw_output 写入；5 unit PASS（4 路径 + retained_unaskable 写回） |
| F11 | 条件路由：diagnose_router | [x] | 2026-05-12 | 纯函数：need_exam 且 exam_round<MAX → recommend_exam；否则 → safety_gate；5 unit PASS |
| F12 | 节点 ⑪：safety_gate | [x] | 2026-05-12 | 规则层（allergy/pregnancy 直接抽取，spec §4.1.2 ⑪ TODO 重构方向待 B 阶段药品规则表落地）+ LLM 兜底（高安全级，失败保守追加通用警告）；4 unit PASS |
| F13 | 节点 ⑫⑬：建议与输出 | [x] | 2026-05-12 | ⑫ generate_advice（failure_reason 三类对应 risk_warnings 提示）+ ⑬ format_response（失败兜底静态模板）；5 unit PASS |
| F14 | StateGraph 编排 | [x] | 2026-05-12 | `src/agent/graph.py` 注册 16 节点 + 2 条件边；`build_app()` 默认 InMemorySaver；2 unit PASS（节点全 + compile 通过） |
| F15 | 全工作流集成测试 | [x] | 2026-05-12 | 2 集成测试：正常 confirmed 全链路 + followup 触顶兜底全链路；interrupt resume 路径留 J 阶段 |

### 阶段 G：API 层与权限系统

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| G1 | FastAPI 应用骨架 | [x] | 2026-05-12 | `src/api/app.py::create_app()` 工厂 + 模块级 `app` uvicorn 入口;`Instrumentator` 接入 `/metrics`(`excluded_handlers=["/healthz", "/readyz", "/metrics"]` 防自污染,§5.2.1 ②);`src/api/routes/__init__.py::register_routers(app)` 集中挂载点(G2-G6 / H8 注释占位,目前 noop);8 unit smoke PASS(/ 404、/metrics 200 + `http_requests_total`/`http_request_duration_seconds` family、/metrics 自身不被采集、/healthz /readyz NOT-IN-SCOPE 锁 404、register_routers idempotent);活 uvicorn 进程 curl 验通 |
| G2 | JWT 认证中间件 | [x] | 2026-05-12 | `src/api/middleware/auth_middleware.py`:bcrypt 5.x `hash_password`/`verify_password` + PyJWT HS256 `encode_access_token`/`decode_access_token` + `get_current_user` Depends + `require_role(*roles)` 工厂(403 守卫,误用空参 ValueError);`src/api/schemas/auth_schema.py` Pydantic `RegisterRequest`(`Literal["patient","admin"]` 防 doctor 误传)/`LoginRequest`/`TokenResponse`/`UserOut`;`src/api/routes/auth.py` 3 端点(`POST /auth/register` 201 + IntegrityError → 409 + `POST /auth/login`(密码错与用户不存在共享 401 防枚举)+ `GET /auth/me` 不打 DB);14 unit(bcrypt salt 不等、verify 异常哈希 False、JWT roundtrip + 过期/伪造/缺 sub 全 401、require_role 三场景)+ 15 integration(端点闭环 + bcrypt 哈希入库验证 + 临时 admin-only 路由验角色守卫)= 29 PASS;活 uvicorn curl 验通 |
| G3 | 限流中间件 | [x] | 2026-05-12 | `src/api/middleware/rate_limiter.py`:`RateLimitBackend` Protocol 抽象配额存储(H6 切 Redis 只换实现)+ `InMemorySlidingWindow`(线程安全,deque[timestamp] 每请求 popleft 过期戳)+ `RateLimitMiddleware`(BaseHTTPMiddleware,超限 429 + `Retry-After` header + JSON body 含 retry_after/limit/window);key 选择:有效 JWT → `user:<sub>`,无效/无 token → `ip:<addr>`(伪造 token 不直接拒,fallback 到 IP 防绕过);excluded_paths 含 `/metrics` + 防御式 `/healthz` `/readyz`(H8);配额走 `settings.api.RATE_LIMIT_PER_MINUTE`(默认 30/分钟);12 unit PASS(窗口满 429 + 时间滑过释放 + key 隔离 + retry_after 单调递减 + JWT 用户隔离 + 伪造 token 落 IP 桶 + /metrics 免限流);活 uvicorn `API_RATE_LIMIT_PER_MINUTE=5` 验通 |
| G4 | 问诊接口 | [x] | 2026-05-20 | `src/api/routes/diagnosis.py` `POST /diagnose` **走 SSE 流式**(`text/event-stream`):首次建 sessions 行 + 走 `graph.astream_events(version="v2")` 监听 `on_chain_start` → 每个节点**进入时**推一条 `{event:"progress", node, text}`(文案查 `_NODE_PROGRESS_TEXT` 15 行 mapping,改用 events 是因为旧 `stream_mode="updates"` 在节点**结束**才 yield,灰字延迟一拍);events 退出后用 `aget_state(...).next` 区分,**4 种 interrupt 状态各自路由**:`initial_ask` 节点内 interrupt 推 `{event:"interrupt", status:"ongoing_initial_ask", pending_questions:[open/history/obstetric]}`(首轮第 1 个 form,⓪a 写 followup_questions/_question)、`wait_followup_answer` 推 `ongoing_followup`(slot 追问 form)、`wait_exam_report` 推 `ongoing_exam`、`analyze_initial_reports` 节点内 interrupt 推 `ongoing_report_upload`(payload 文案在 `INITIAL_REPORT_UPLOAD_PROMPT` 单点维护);终态推 `{event:"completed", session_id, status, final_response, diagnosis_result, medication_advice, risk_warnings}`;generator 内异常推 `{event:"error", session_id, detail}`(SSE 头已发不能 raise);终态写 rag_trace/conversations 时机和 §9.6.2/§9.6.5 裸代码模板完全一致(generator 最后一个 yield 之前);后续轮 `Command(resume=...)` 恢复 interrupt 同源;失败兜底 `_build_error_info` 派生 step;`patient_repo.load_medical_history` 真查询 8 张表 → spec §4.1.1 dict;10 integration PASS(mock astream_events + aget_state,helper `_read_sse_events` 解 SSE,验 progress 推送 / 首轮 completed / 失败兜底 / 四种 interrupt(initial_ask/followup/exam/report_upload) / 越权 403 / 不存在 404)。**首轮 LLM 调用从 2 次(原 ⑦+①)合并为 1 次(①),节省 ~3s 等待**;⑦ 退化为"⑥ 后追问解析专属",intake_router 简化为 2 路。⚠️ checkpointer 当前 `InMemorySaver` 模块级单例,生产应换 `PostgresSaver`(deps 已有);`DiagnoseResponse` Pydantic 模型保留在 schemas 仅作字段定义参考,实际接口不再返 JSON。**2026-05-23 X3/X4**:加 `POST /diagnose/upload` 接收 multipart 文件(15MB 上限 / MIME 白名单 image\*+pdf / 路径隔离 `/tmp/uploads/<sid>/`),返回 `file_ref` 给前端;①.5 与 ⑧b **共享 `exam_results` 字段**(LangGraph 按 snapshot.next 自然分流),前端首诊和检查回传都走分框上传(每框=一份独立报告,框内 multiple 多文件,标签选填)|
| G5 | 患者信息接口 | [x] | 2026-05-13 | `src/api/routes/patient.py` 11 端点(GET/PUT `/me` + 三张 ⚠️必问表 medical-history/allergies/medications POST/DELETE);全部 `require_role("patient")`,**没有 path 参数**(身份隔离从设计上根除越权);PUT `/me` 自动建 patients 行(注册时不建,延后到首次填档);`_delete_owned_or_404` 验 owner 防泄漏存在性;`_ensure_patient_row` 子表写入前兜底建 patients 行;8 integration PASS(admin 角色 403 / 部分更新语义 / 创建-查看-删除闭环 / 跨用户 404 / safety_gate 同链路读取一致)。⚠️ 5 张子表(surgical_trauma/transfusion/family_history/menstrual/exam_reports)只在 GET 全档案返回,独立 CRUD 留 TODO;exam_reports 文件上传需 multipart endpoint 单独做 |
| G6 | 管理员接口 | [x] | 2026-05-13 | `src/api/routes/admin.py` 6 端点(GET `/users` 分页 + DELETE `/users/{id}`(防删自己)+ system_config GET 列表/单条/PUT/DELETE + `/kb/upload` stub);全部 `require_role("admin")`;PUT `/admin/config/{key}` 同事务写 system_config + config_change_log(spec §5.3.1 末);`change_reason` 字段 schema 强制 `min_length=1` 防止变更无说明;10 integration PASS(角色守卫 + 分页 + 防删自己 + change_log 同事务双行 + delete 也写日志 new_value=null + stub 返 202+unimplemented)。⚠️ 知识库上传走 stub:C7 ingestion pipeline 入口函数未做完,本端点提示走 scripts 路径;C7 完工后改 multipart 上传 + 后台 task + kb_change_log |
| G7 | Nginx 反向代理 | [x] | 2026-05-13 | `infra/docker/nginx.conf`(docker-compose 已挂);worker 自动 + upstream keepalive 32 + `client_max_body_size 50M`(为知识库/exam_reports 上传)+ `proxy_read_timeout 600s`(LLM 慢路径)+ `set_real_ip_from + X-Forwarded-For + X-Real-IP`(让 G3 限流拿到真实客户端 IP,否则全是 nginx 容器 IP IP 桶失效)+ `X-Forwarded-Proto`(FastAPI redirect/OpenAPI scheme);`/nginx-health` 内嵌 200 给外部 LB 探活;HTTPS / TLS 留生产部署再接(本期 demo HTTP)。**真起验证**(2026-05-13):`docker-compose.demo.yml` override(J0 `Dockerfile.api` 完工前用,host 跑 uvicorn 8000 + nginx 容器 `extra_hosts: api:host-gateway` 把 nginx.conf 里 `server api:8000` 解到宿主机,主 compose 不动 J0 无缝继承)— curl `localhost:80/{nginx-health,/metrics,/auth/me,/auth/register,/auth/me with token}` 全链路 200 / 401 闭环验通,nginx 日志 rt=2ms,X-Real-IP 透传 OK |

### 阶段 H：基础设施增强

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| H1 | Redis 缓存客户端 | [x] | 2026-05-14 | `src/db/redis/cache.py` Cache-Aside；`config:<key>` 命名空间 + 60s TTL（settings.redis.CONFIG_CACHE_TTL）；`get_redis_client()` lru_cache 单例 + 2s 超时；Redis 不可用 / GET / SETEX 任一抛 → WARNING + 回源 loader（**对调用方完全透明**，spec §5.1 降级模式）；JSON 损坏自愈（DEL + 回源）；`loader()` 返 None 不污染缓存；redis 命令延迟同时上报 `redis_command_latency_seconds`（H2）；10 unit PASS |
| H2 | Prometheus 指标埋点 | [x] | 2026-05-14 | `src/common/metrics.py` 扩 4 个上下文指标（`context_tokens_per_llm_call` / `context_structured_fields_size` / `context_messages_count` / `context_loop_iterations`，spec §4.2.7）+ 5 个依赖层指标（`db_query_latency_seconds` / `db_pool_checkedout` / `redis_command_latency_seconds` / `milvus_rpc_latency_seconds` / `milvus_rpc_errors_total`，spec §5.2.1 ③）；新增 `src/db/postgres/metrics.py` 订阅 SQLAlchemy `before/after_cursor_execute` 自动产 `db_query_latency_seconds{operation}`，`install_engine_metrics()` 幂等；`infra/prometheus/prometheus.yml` 配 6 个 scrape job（api / prometheus / node-exporter / dcgm-exporter / loki / grafana）；9 unit PASS（指标对象/label/SQLAlchemy event/`/metrics` 端点能拉到所有新增指标） |
| H3 | Grafana 仪表盘 | [x] | 2026-05-14 | `infra/grafana/datasources/datasources.yml` 自动 provision Prometheus + Loki 两个 datasource（Loki derivedFields 配 trace_id 跳转，对齐 §5.2.1.1 末"日志↔审计联通"）；`infra/grafana/dashboards/dashboards.yml` provider；两个 dashboard JSON：① 应用性能（QPS / 5xx / HTTP P95 / LLM 延迟 P50P95 / 失败重试兜底 / diagnose 失败原因 / context tokens / 循环轮次 / 依赖层 PG·Redis·Milvus）② 硬件资源（CPU / 内存 / 磁盘 / 网络 / GPU 使用率·显存·温度·功耗）；`docker-compose.yml` 补挂 datasources 卷 + 增加 grafana → loki depends_on |
| H4 | 日志采集（Promtail → Loki） | [x] | 2026-05-14 | `config/logging_config.py` 用 stdlib json 自写 `_MedicalJsonFormatter`（spec §5.2.1.1 强制字段：`timestamp`/`level`/`logger`/`trace_id`/`session_id`/`patient_id`/`node`/`message`/`exc_info`，level 大写、ISO8601-Z timestamp、`extra={}` 透传）；`_ContextVarFilter` 把 ContextVar 注入 `LogRecord`；`configure_logging()` 替换 root + uvicorn handler 幂等；`src/api/middleware/trace_id.py` 入口生成/复用 UUID4，回写 `X-Trace-Id` header；`src/api/app.py` 在 RateLimitMiddleware 之后挂（更外层，让 429 也带 trace_id）；**spec 偏差**：spec 提示用 `python-json-logger`，本实现改用 stdlib（避免 cu128 索引策略冲突），输出字段完全对齐；`infra/promtail/promtail-config.yml` 抓 docker json-file driver stdout，二级 JSON 解析提取 level/node 做 label，trace_id 留正文（避免 Loki 高基数）；`infra/loki/loki-config.yml` 单机 filesystem 存储、保留 90d（对齐 §5.2.3.5）；17 unit PASS |
| H5 | Node Exporter 硬件监控 | [x] | 2026-05-14 | `docker-compose.yml` 早就配好（`prom/node-exporter:latest`，挂 /proc /sys / 监控宿主机），`docker compose up -d node-exporter` 起来 → `docker exec node-exporter wget -qO- /metrics` 验通拿到 `go_*` 等基础指标；scrape job 加在 prometheus.yml |
| H5b | DCGM Exporter GPU 监控 | [~] | 2026-05-14 | docker-compose 配置已有；起容器报 "could not select device driver 'nvidia' with capabilities: [[gpu]]" — 宿主机缺 `nvidia-container-toolkit`，需用 `apt install nvidia-container-toolkit && systemctl restart docker` 装上才能跑 |
| H6 | Redis 缓存与业务层对接 | [x] | 2026-05-14 | `src/db/redis/rate_limit_backend.py` `RedisSlidingWindow` Lua 脚本原子 ZSET 滑动窗口（与 G3 InMemorySlidingWindow 语义对等可热替换；NOSCRIPT 自愈、Redis 不可用 fail-open）；`src/api/middleware/rate_limiter.py` 默认 backend 改为 Redis 版（构造时若不显式传 backend 自动用 Redis）；G3 测试 `_make_app` 显式传 InMemorySlidingWindow 避免串库；动态配置在 H7 走 H1 cache（Cache-Aside 60s 自动生效）；7 unit PASS |
| H7 | 动态配置管理 | [x] | 2026-05-14 | `src/db/postgres/system_config.py`：`get_dynamic_config(key, default)` 经 H1 cache 读 → 未命中回源 `system_config` 表 → 写回 60s TTL；`set_dynamic_config(key, val, operator_id, ...)` **同事务**写 system_config + ConfigChangeLog（spec §5.3 末），提交后 `invalidate_config()` 失效缓存让全节点最多 60s 切新值；`list_dynamic_configs()` 直读 PG 不走缓存（admin UI 看实时）；`_infer_value_type` 自动推断 `BOOL`(在 INT 之前判)/`INT`/`FLOAT`/`STRING`/`JSON`；docstring 反复提示 `agent_limits` 六常量等不进本表(spec §5.3 分界)；9 unit PASS |
| H8 | 健康检查端点 | [x] | 2026-05-14 | `src/api/routes/health.py`：`GET /healthz` 零依赖固定 200（即便 PG 挂了也返）；`GET /readyz` `asyncio.wait_for + run_in_executor` 并发探 PG `SELECT 1` + Milvus `has_connection + get_server_version`，2s 超时；任一失败 503 + `failing` 列表；Redis 不可用仍算 ready（spec §5.2.4 / §5.1 降级模式）；docker-compose api 容器 healthcheck 从原 `/health`（不存在）改正为 `/healthz`；nginx 自身 `/nginx-health` 已在；G1 留的 "test_healthz_and_readyz_not_implemented_yet" 占位测试同步删；8 unit PASS |

### 阶段 I：评估体系

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| I1 | RAG 离线评估 | [ ] | | |
| I2 | Agent 离线评估（L1~L5 梯度） | [ ] | | |
| I3 | LLM Judge | [ ] | | |
| I4 | 在线追踪 | [ ] | | |
| I5 | 评估脚本入口 | [ ] | | |

### 阶段 J：端到端验收与文档收口

| 编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|------|---------|------|---------|------|
| J0 | Docker 化部署 | [x] | 2026-05-14 | `infra/docker/Dockerfile.api`(`uv:python3.12-bookworm-slim` base + `uv sync --frozen` + uvicorn `--proxy-headers`,layer cache 友好,镜像 12GB);`.dockerignore` 排除 `.venv` / `tests/` / `data/` / `infra/{grafana,prometheus,loki,promtail}`(避免 5GB+ 上下文);`docker-compose.yml` api service 加 `environment` 覆盖 `POSTGRES_HOST=postgres` / `MILVUS_HOST=milvus-standalone` / `REDIS_URL=redis://redis:6379`(.env 默认 localhost 是给 host uvicorn 的)+ `deploy.resources.reservations.devices` 透传 GPU(配合 nvidia-container-toolkit);`docker compose up -d api nginx` → api healthcheck `/healthz` 60s 内转绿;端到端验证(2026-05-14):nginx 80 → api 8000 全链路,`/healthz` `/readyz` 200,响应 header 带 `x-trace-id`(H4 中间件工作);Prometheus 6/6 target up(api / dcgm-exporter / grafana / loki / node-exporter / prometheus);触发 30 次 `/readyz` 后 PG `db_query_latency_seconds_count{operation="SELECT"} = 31`(SQLAlchemy event listener H2);api 容器 stdout JSON 日志被 Promtail 推到 Loki,query `{job="docker"} \|= "trace_id"` 命中;`/metrics` 暴露全部 11 个业务指标 family(structured_output_* / context_* / db_query / redis_command / milvus_rpc / diagnose_failure);LLM 相关指标暂时无值,等真跑 `/diagnose` 后会自动出 |
| J1 | E2E：Ingestion 全链路 | [ ] | | |
| J2 | E2E：Retrieval 全链路 | [ ] | | |
| J3 | E2E：Agent 全链路 | [ ] | | |
| J4 | E2E：API 接口 | [ ] | | |
| J5 | README 完善 | [ ] | | |
| J6 | 清理与一致性检查 | [ ] | | |






