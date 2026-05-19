# 2 技术选型：
本项目选型使用大模型的时间为2026/3/6
## 2.1 Embedding 模型选型

Embedding 模型负责将文本转换为向量，用于粗排召回。医学场景对 Embedding 精度的要求远高于通用场景：术语歧义多（"MI"可指心肌梗死或二尖瓣关闭不全）、近义表达丰富（患者口语 vs 临床术语）、细粒度语义区分关键（"左心衰"vs"右心衰"、"急性"vs"慢性"）。参数量不足的模型无法在有限维度的向量空间中编码这些细微差异，导致语义塌缩——即把临床上不同的概念压成接近的向量，直接损害召回质量。

### 选型结论：Qwen3-Embedding-8B（Qwen/Qwen3-Embedding-8B），部署于 GPU

#### 选型理由

1. **8B 参数量，医学语义容量充足**：8B 参数基于 Qwen LLM 架构（decoder-based），相比传统 encoder 模型（如 BGE-M3 的 568M），语义编码容量提升一个数量级。更大的参数量意味着模型能在向量空间中为医学术语的细粒度差异分配足够的表征空间，避免语义塌缩。C-MTEB 总分 73.84，远超 BGE-M3 的 ~66-67。

2. **Qwen 底座，中文医疗预训练充分**：Qwen 系列在中文语料（含医学文献、临床指南、药品说明书等）上的预训练规模远超 BAAI 的 encoder 系列模型。与本项目 LLM 选型（Qwen 系列）同源，tokenizer 一致，对中文医学术语的分词方式相同，粗排→精排之间语义对齐更好。

3. **超长上下文支持（32,768 tokens）**：医疗指南的 Chunk 可能较长，32K 上下文窗口能完整编码长段落语义，无需担心截断丢失关键信息。

4. **高维向量（4096 维）**：输出 4096 维 Dense 向量，向量空间容量大，为医学术语的精细区分提供更充裕的表征维度。本项目的 Collection Schema（见 2.4.1）中 `dense_vector` 字段需相应调整为 4096 维。

5. **Sparse 路由不受影响**：本项目的 Sparse 检索采用 Milvus 内置 BM25 全文检索（见 3.2.2），与 Embedding 模型无关，Qwen3-Embedding-8B 专注 Dense 编码，职责清晰。


虽然本项目已有多向量表示（original + summary + question）、混合检索（Dense + BM25）、Reranker 精排等多层机制提升召回质量，但这些机制**无法弥补粗排阶段的根本性召回缺失**——如果 Embedding 模型因语义容量不足而未能将正确文档召回到候选集中，后续的 Reranker 和融合策略再强也无从精排。在医学场景中，漏召一篇关键指南可能直接影响诊断建议的完整性和安全性，因此 Embedding 环节值得投入更大参数量的模型。

#### 部署策略：GPU 推理（RTX 5070 Ti 16GB）

LLM 推理通过云端 API 调用（见 2.2），GPU 显存全部分配给 Embedding 和 Reranker，部署策略如下：

- **INT8 量化部署**：8B 模型 INT8 量化后显存占用约 8.5-8.8GB，RTX 5070 Ti 16GB 可轻松容纳，剩余显存供 Reranker（INT8 约 2.6GB）使用。
- **离线 Embedding（文档入库）**：GPU 推理速度远快于 CPU，批量入库效率大幅提升，无需安排夜间错峰执行。
- **在线 Query Embedding（实时查询）**：GPU 推理延迟极低（单条 Query 通常 <10ms），用户体验优于 CPU 方案。
- **与 Reranker 共享 GPU**：Embedding 和 Reranker 负载天然错峰（Embedding 在入库时批量执行，Reranker 在查询时实时执行），可共享 GPU 资源，无需额外硬件。

#### 其他候选模型排除理由

| 候选模型 | 排除原因 |
|---------|---------|
| **BGE-M3（568M）** | 参数量不足，医学场景语义编码能力有限，C-MTEB 检索分数落后约 10 个点 |
| **Conan-embedding-v2（1.48B）** | 传统 encoder 架构，参数量虽大于 BGE-M3 但仍有限；腾讯生态，社区资源和文档不如 Qwen；C-MTEB 检索分 78.31 虽高但为 encoder 天花板 |
| **Qwen3-Embedding-4B** | C-MTEB 72.26，与 8B 差 1.5 分；显存节省有限（INT8 约 4-5GB vs 8-10GB），在 16GB 显卡上无需为省这点显存牺牲模型能力 |
| **Qwen3-Embedding-0.6B** | 参数量与 BGE-M3 相当（0.6B vs 0.568B），C-MTEB 66.33 无本质提升，不解决核心问题 |
| **Seed1.6-Embedding（字节）** | API only，无法本地部署；入库批量调用成本累积；API 模型升级后向量不兼容需全量重索引；医疗数据外传存在合规风险 |
| **云端 Embedding API（通用）** | 同上，且引入网络延迟和外部依赖；Embedding 需要与 Reranker 共享 GPU，本地部署更简洁高效 |

## 2.2 Agent 及 RAG 系统模型选型（云端 API）

### 2.2.1 选型结论：Qwen 系列云端 API（阿里云 DashScope）

**选型结论：`qwen-max`（首选）/ `qwen-plus`（备选）**，通过阿里云 DashScope OpenAI-compatible 接口调用。

#### 选型理由

**1. 与 Embedding 模型同族系，构成完整 Qwen 生态**

本项目 Embedding 选型为 Qwen3-Embedding-8B（见 2.1），与 Qwen 系列 LLM 共享以下底层一致性：

- **Tokenizer 完全相同**：Qwen 全系列使用同一套 tiktoken BPE 分词器。医学术语（如"氨氯地平片"、"急性心肌梗死"）在 Embedding 阶段和 LLM 推理阶段的分词结果完全一致，避免跨模型族系时 token 边界不对齐的问题。
- **预训练数据对齐**：Embedding 模型和 LLM 在相同的中文医疗语料（包括医学文献、临床指南、药典等）上预训练，二者对同一医学概念的"理解"处于同一语义空间。RAG 检索回来的 chunk 直接注入 LLM context，语义摩擦极小，LLM 能高效利用检索内容。
- **指令式 Embedding 对齐**：Qwen3-Embedding-8B 支持 task instruction（如 `"Represent this medical query for retrieval:"`），可让 Embedding 的向量表征方向与 Qwen LLM 的 query 理解方式进一步对齐，提升粗排召回的相关性。

**2. GPU 显存全部分配给 Embedding 和 Reranker**

本项目硬件为 RTX 5070 Ti（16GB 显存）。LLM 通过云端 API 调用，GPU 显存全部释放给 Qwen3-Embedding-8B（INT8 约 8.5-8.8GB）和 BGE-Reranker-v2-minicpm-layerwise（INT8 约 2.6GB），合计约 11.1-11.4GB，留有充足余量应对推理激活值与显存碎片，检索质量和精排速度大幅提升。

**3. 接口完全 OpenAI-compatible，代码简洁**

DashScope 提供 OpenAI-compatible 接口，通过环境变量配置即可切换模型，`src/models/llm_client.py` 业务代码无需改动。

#### 云端 Qwen 模型对比

| 模型 | 定位 | 适用场景 |
|---|---|---|
| **qwen-max**（首选） | 旗舰，推理能力最强 | 复杂诊断推理、多轮追问、用药安全判断；支持 thinking 模式 |
| **qwen-plus**（备选） | 均衡，成本低约 60% | 成本敏感场景，常见问诊、症状分析 |
| qwen-turbo | 极速低成本 | 不适合本场景，医疗推理质量不足 |

#### 接口配置

```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-xxx
LLM_MODEL_NAME=qwen-max
```

### 2.2.2 选型测评结果

具体的测评方式在第 6 部分有详细介绍，本节陈述最终结果。

云端 `qwen-max` 直接采用，无需本地测评流程。若后续成本压力较大，可在 `qwen-max` 与 `qwen-plus` 之间通过以下维度进行业务场景对比：诊断准确率、追问合理性、用药安全判断、结构化输出稳定性。

> 在本项目中，作者没有高质量的标注数据，微调暂不计划。厂商已做过 DPO/RLHF 对齐；若出现持续的有害输出或不期望行为，可考虑少量 DPO 对齐。



## 2.3 reranker模型选型
Rerank 模型则是对召回的候选文档做精排。通常是 cross-encoder 架构——将 query 和 document 拼接在一起输入模型，直接输出一个相关性分数。因为 query 和 document 之间有充分的交互注意力，所以精度更高，但计算成本也更大，不适合直接用于全量检索，只适合对少量候选（比如 top-20 到 top-100）重新排序。

### 选型结论：BGE-Reranker-v2-minicpm-layerwise（BAAI/bge-reranker-v2-minicpm-layerwise），部署于 GPU

#### 选型理由

1. **中文医疗场景精度最高**：基座为 MiniCPM-2B（清华 & 面壁智能），中文预训练语料充分，对 ICD-10、SNOMED CT 等医学术语的理解能力显著优于 XLM-RoBERTa 基座的 v2-m3。在 C-MTEB Reranking 中文子集上得分明显领先同系列其他型号。同时支持中英双语，可覆盖英文医学文献的精排需求。

2. **Cross-Encoder 架构，2.4B 参数量精排精度高**：作为 cross-encoder，将 query 和 document 拼接后做全注意力交互，精度远高于 Embedding 的 bi-encoder 相似度。2.4B 参数量相比 v2-m3 的 568M 提升一个量级，语义判别能力更强。即使粗排（Embedding）和精排（Reranker）来自不同模型族系（Qwen vs BAAI），cross-encoder 的全交互机制天然弥补了这一差异——Reranker 独立判断 query-doc 相关性，不依赖粗排阶段的向量表征。

3. **Layerwise 推理，精度与速度连续可调**：模型在每一层都训练了分类头，可通过 `cutoff_layers` 参数选择从第 N 层提前提取分数，而非必须跑完全部 28 层。全层推理获得最高精度；截断至前 20 层可在精度损失极小的情况下提速约 30%。这为生产环境提供了灵活的精度-延迟调节旋钮。

4. **长上下文支持（8192 tokens）**：医疗指南的 Chunk 可能较长，8192 token 窗口确保 query-document 对的完整交互，不会因截断丢失精排信息。

#### 部署策略：GPU 推理，与 Embedding 模型共享显卡

本项目 LLM 迁移至云端 API 后（见 2.2），RTX 5070 Ti 16GB 显存由 Embedding 模型（Qwen3-Embedding-8B，INT8 约 8.5-8.8GB）和 Reranker 共享：

- **二者不会同时高负载**：Embedding 在文档入库时批量执行，Reranker 在用户查询时实时执行，负载天然错峰。
- **INT8 量化部署，显存安全**：2.4B 参数 INT8 量化后约 2.6GB，与 Embedding 模型合计约 11.1-11.4GB，16GB 显卡余量约 4.6-4.9GB，充分覆盖推理激活值（~0.5-1GB）、CUDA 固定开销（~0.5-0.8GB）和显存碎片。采用 FP16（~4.8GB）会导致合计 13.3-13.6GB，在双模型同时推理时存在 OOM 风险，因此不采用。
- **INT8 对精排精度影响极小**：Reranker 输出的是用于排序的相对分数而非生成文本，对量化精度损失不敏感。
- **GPU 推理速度优秀**：20 个 query-doc pair 精排延迟约 40-80ms（全层），使用 layerwise 截断可进一步降至 30-60ms，用户体验良好。
- **候选量有限**：精排仅处理 RRF 融合后的 Top-20 候选（见 3.2.3），不涉及大批量计算。

#### 备选方案与排除理由

| 备选模型 | 排除原因 |
|---------|---------|
| **Cohere Rerank** | 闭源云端 API，医疗数据外传存在合规风险；引入外部依赖影响系统稳定性 |
| **LLM Rerank（Qwen 自身做精排）** | 会抢占推理模型的 GPU 资源和推理队列，增加端到端延迟；结构化输出不如 cross-encoder 稳定；成本高于专用 Reranker |
| **BGE-Reranker-v2-m3** | 568M 参数量，精排精度低于 v2-minicpm-layerwise；项目显存充足（剩余 5-6GB），无需为节省显存牺牲精度 |
| **BGE-Reranker-v2-gemma** | 基于 Gemma 2B，英文预训练为主，中文医疗术语理解不如 MiniCPM 基座；无 layerwise 灵活性 |
| **BGE-Reranker-large（v1）** | 旧版本，中文能力和长上下文支持不如 v2 系列，最大输入仅 512 tokens，无法覆盖本项目的长 Chunk 场景 |

#### 与系统架构的衔接

- **输入**：RRF 融合 + 多向量聚合后的 Top-M 候选（见 3.2.2），每条候选为 [query, original_content] 对
- **输出**：相关性分数排序后的 Top-K 结果，传给 LLM 生成诊断
- **回退机制**：Reranker 超时或不可用时，直接返回 RRF Top-K，确保系统可用性（见 3.2.3 回退策略）
- **Layerwise 配置**：生产环境默认使用全层推理（28 层）以获得最高精度；可通过配置 `cutoff_layers` 参数在延迟敏感场景下切换至截断模式

## 2.4 数据存储选型及具体设计：
### 2.4.1. 原始文档向量化的向量库：Milvus

每个 Chunk 在 Milvus 中对应 4~5 条向量记录（1 original + 1 summary + 2~3 question）：

| vector_type | id 规则 | Dense (Qwen3-Embedding-8B) | BM25 全文检索 | 说明 |
|-------------|---------|:-----:|:------:|------|
| `original` | `{chunk_id}` | ✅ | ✅ | 原文向量 + 全文索引，支持语义检索与关键词检索 |
| `summary` | `{chunk_id}_summary` | ✅ | ❌ | 摘要向量，提升对模糊 query 的匹配能力 |
| `question` | `{chunk_id}_q{n}` | ✅ | ❌ | 问题向量，弥合患者口语与临床文本的语义鸿沟 |

summary / question 记录不参与 BM25 全文检索——关键词匹配应基于原文，而非 LLM 改写文本，避免语义漂移。BM25 由 Milvus 2.4+ 内置全文检索引擎承担，基于 `original_content` 字段建立倒排索引，无需 Embedding 模型输出 Sparse 向量。

**Milvus Collection Schema**：

```
{
    "id":               str,             # 本条记录唯一 ID（见上表）
    "source_chunk_id":  str,             # 所属原始 chunk_id（original 记录与 id 相同）
    "vector_type":      str,             # "original" | "summary" | "question"
    "dense_vector":     List[float],     # Qwen3-Embedding-8B 语义向量，4096 维（所有记录均有）
    "text_for_bm25":    str,             # BM25 全文检索字段（仅 original 有值，summary/question 存空串；Milvus 2.4+ 自动建立倒排索引）
    "original_content": str,             # 原始 chunk 文本，冗余存储，命中后无需回查 PostgreSQL
    "source_id":        str              # Pre-filter 字段：按来源文档过滤（见 2.4.2 sources 表）
}
```

`title`、`heading_path` 等展示字段不在 Milvus 冗余，检索命中后以 `source_chunk_id` 回查 PostgreSQL `chunks` 表获取。


 ### 2.4.2. 元数据存储：PostgreSQL

PostgreSQL 负责存储所有 Chunk 的结构性元数据与增强元数据，支撑幂等写入、僵尸清理、增量 Embedding 判断及检索结果的上下文还原。向量数据本身存储于 Milvus，PostgreSQL 不存储向量。

**sources 表**（来源文档注册表，source_id 的权威来源）

```sql
sources (
  source_id    TEXT PRIMARY KEY,          -- 文档唯一 ID（见 3.1.4.1）
  file_name    TEXT NOT NULL,             -- 原始文件名
  file_path    TEXT,                      -- 文件存储路径
  doc_type     VARCHAR(50),               -- 文档类型，如 guideline / textbook / protocol
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
)
```

**chunks 表**（Chunk 元数据核心表）

```sql
chunks (
  -- 幂等性字段（见 3.1.4）
  chunk_id              TEXT PRIMARY KEY,   -- SHA256(source_id:heading_path_id:relative_chunk_index)
  source_id             TEXT NOT NULL REFERENCES sources(source_id),
  heading_path_id       TEXT NOT NULL,      -- SHA256(H1_id:H2_id:...) 标题路径哈希
  heading_path          TEXT NOT NULL,      -- 人类可读标题路径，如 "第2章 > 2.1 > 3.1.4"，用于检索结果展示
  relative_chunk_index  TEXT NOT NULL,      -- 同标题路径下的块序号；子块用 "0/1/2..."，父块用 "parent"，图表 chunk 用 "{chunk_type}:p{page_idx}_b{block_idx}"（见 §3.1.4.2 步骤 3 图表 chunk 约定）
  parent_chunk_id       TEXT REFERENCES chunks(chunk_id),
                                            -- 父块 ID（Small-to-Big 父子索引，见 3.1.2）；NULL 表示本块即为顶层父块
  chunk_type            VARCHAR(20) NOT NULL DEFAULT 'child',
                                            -- parent / child / table / figure
                                            -- figure 涵盖所有"以图片为主"的源(mineru type=image 的 flowchart 子集 + type=chart 全部;chart 识别质量差，统一按图片处理)
                                            -- 详见 3.1.2 图表/影像处理章节
  image_path            TEXT,               -- 图表截图相对路径(table / figure chunk 用)；NULL 表示非图表 chunk
  sub_type              VARCHAR(20),        -- mineru sub_type(figure chunk 用，记录 'flowchart' 或原 chart 子类 'line'/'bar' 等便于回溯)；NULL 表示非 image/chart 来源 chunk
  chunk_raw_text        TEXT NOT NULL,      -- Chunk 原始文本：child=正文段落；table=caption + html + footnote(html 高质量保留)；figure=caption + footnote(不写入 mineru mermaid / markdown，质量太差从不消费，详见 §3.1.2)；parent=合并子块文本(仅用于 child 命中后 Small-to-Big 展开父级 context，**不入向量也不入 BM25**)。child / table / figure 走 BM25 sparse；child 同时作为 dense `original` 向量来源(table / figure 的 dense `original` 走 medical_statement)
  medical_statement     TEXT,               -- table / figure 专用：LLM 生成的 100-300 字医学陈述(把图表数据线性化为陈述句)。**table / figure 的 dense `original` 向量来源**(替代 chunk_raw_text，因 figure 的 chunk_raw_text 只有 caption 不够 dense embed、table 虽有 html 但作 dense 原文表达力不足，见 §3.1.2)；child / parent 此列为 NULL
  content_hash          TEXT NOT NULL,      -- 变动检测信号（见 3.1.4.3）：child / parent = SHA256(chunk_raw_text)；table / figure = SHA256(chunk_raw_text + "\n" + medical_statement)，两路来源(html/caption 与 LLM 陈述)任一变化即触发重新 embed

  -- LLM 增强字段（见 3.1.3）
  title                 TEXT,              -- LLM 生成的精准小标题
  summary               TEXT,             -- LLM 生成的内容摘要，同时作为摘要向量文本来源（见 3.1.5）
  hypothetical_questions TEXT[],          -- LLM 生成的假设性问题数组（3 条，见 3.1.5）

  -- 运维状态字段
  embedding_status      VARCHAR(20) NOT NULL DEFAULT 'pending',
                                          -- pending / done / failed / skip
                                          -- pending：待 Embedding；done：向量已写入 Milvus；failed：Milvus 写入失败待补偿
                                          -- skip：父块专用，永不向量化(见 3.1.2 父子索引)
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
)
```

**索引**

```sql
CREATE INDEX idx_chunks_source_id        ON chunks (source_id);           -- 僵尸清理差集查询
CREATE INDEX idx_chunks_heading_path_id  ON chunks (heading_path_id);     -- 按标题路径聚合查询
CREATE INDEX idx_chunks_content_hash     ON chunks (content_hash);        -- 跨文档内容去重
CREATE INDEX idx_chunks_embedding_status ON chunks (embedding_status)     -- 增量 Embedding 任务扫描
  WHERE embedding_status NOT IN ('done', 'skip');
CREATE INDEX idx_chunks_parent_chunk_id  ON chunks (parent_chunk_id)      -- 僵尸清理按父块分拣子块（见 3.1.4.3）
  WHERE parent_chunk_id IS NOT NULL;
CREATE INDEX idx_chunks_chunk_type       ON chunks (chunk_type);          -- 按 chunk 类型聚合查询(图表/正文等)
```

> `heading_path`（明文）与 `heading_path_id`（哈希）同时存储：后者用于 chunk_id 推导，前者用于检索结果展示来源标题，职责不同，不可合并。

### 2.4.3. 对话与会话记录：PostgreSQL

**sessions 表**（会话管理，串联同一患者的一次完整问诊过程）

```sql
sessions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES users(id),
  title         TEXT,                    -- 会话标题（可由 LLM 自动生成摘要）
  status        VARCHAR(20) NOT NULL DEFAULT 'active',  -- active / closed / archived
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)
```

**索引**

```sql
CREATE INDEX idx_sessions_user      ON sessions (user_id, created_at DESC);   -- 按用户查历史会话
CREATE INDEX idx_sessions_status    ON sessions (status) WHERE status = 'active';  -- 查活跃会话
```

**conversations 表**（对话记录，每条代表一次用户-系统交互）

```sql
conversations (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id    UUID NOT NULL REFERENCES sessions(id),   -- 所属会话
  user_id       UUID NOT NULL REFERENCES users(id),      -- 冗余存储，避免跨表 JOIN
  user_input    TEXT NOT NULL,            -- 用户原始输入
  llm_output    TEXT NOT NULL,            -- LLM 回复
  rag_context   JSONB,                   -- 本轮检索上下文快照（chunk_id 列表 + 分数）
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)
```

**索引**

```sql
CREATE INDEX idx_conversations_session ON conversations (session_id, created_at);  -- 按会话查对话流
CREATE INDEX idx_conversations_user    ON conversations (user_id, created_at DESC); -- 按用户查历史
```
### 2.4.4. 原始指南/教材文档存储：PostgreSQL `raw_documents` 表

PostgreSQL `raw_documents` 表负责存储 MinerU 解析后的所有原始产物，以 `source_id` 为主键，与 `sources` 表（2.4.2）一一对应。

**存储动机**：MinerU 产物既有深度嵌套 JSON（`content_list`、`middle`），又有长文本 Markdown，结构异构且以"写一次、按需读"为主要访问模式。将其与 `sources`、`chunks` 合并到同一 PostgreSQL 库中：(1) `jsonb` 字段（GIN 索引可用）满足 schema 异构容纳需求，等价于文档数据库的灵活性；(2) 长文本走 `text` 字段，PostgreSQL 自动 TOAST 行外存储，性能与文档数据库无差；(3) 与 `sources` 同库后获得跨表 ACID 事务，避免"sources 写成功、原始产物写失败"的双写补偿问题；(4) 减少一项独立服务的运维与连接池负担。

**PostgreSQL 表：`raw_documents`**

```sql
raw_documents (
  source_id        TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE CASCADE,
                                          -- 主键 + 外键双重身份：与 sources 表 1:1，删源文档时级联清理
  file_name        TEXT NOT NULL,         -- 原始文件名，如 "2024心力衰竭指南.pdf"
  stored_at        TIMESTAMPTZ NOT NULL DEFAULT now(),  -- 本条记录写入时间

  -- ── MinerU 文本产物 ──────────────────────────────────────────────
  markdown_content TEXT NOT NULL,         -- target_document.md 全文，供 chunking pipeline 直接读取

  -- ── MinerU JSON 产物（jsonb 原样存入，不做二次解析）─────────────
  content_list     JSONB NOT NULL,        -- target_document_content_list_v2.json(mineru 2.x 推荐格式)
                                          -- 顶层 list[页数],每页 list[block],block = {type, content, bbox};
                                          -- 真实嵌套结构与 10 种 block.type 见 §2.4.4.1
                                          -- 主要消费者:chunking 阶段(§3.1.2)按白名单抽正文、识别表格做双粒度处理
                                          -- 注:mineru 也输出 v1 格式 content_list.json(扁平 list,带 page_idx),向后兼容用,本项目不消费
  middle_data      JSONB NOT NULL,        -- target_document_middle.json
                                          -- 含 token 级版面分析结构,体积大(典型 16-84MB,极端 300MB+,PG TOAST 自动行外存储)
                                          -- 主要用途:排查解析异常(如 OCR 漏字、表格识别错位时翻 token bbox 定位)
  model_data       JSONB NOT NULL,        -- target_document_model.json
                                          -- 模型推理细节(typical 2-19MB),想看 mineru 模型对某块的 layout 分类置信度时翻它

  -- ── 原始文件引用 ─────────────────────────────────────────────────
  pdf_path         TEXT NOT NULL          -- 原始 PDF 在本地磁盘的绝对路径，文件本身不入库
)
```

**索引**

```sql
-- 主键索引由 PRIMARY KEY 自动建立，无需重复声明
-- GIN 索引：支持 content_list 内 type 字段聚合查询（如按"表格块/图像块"过滤）
CREATE INDEX idx_raw_documents_content_list_gin ON raw_documents USING GIN (content_list);
```

**字段说明**

| 字段 | 来源 | 主要用途 |
|------|------|---------|
| `markdown_content` | `target_document.md` | 渲染产物(同信息以 markdown 文本形式呈现),保留作为 raw 备份与版面追溯辅助。**chunking 不消费此字段**,所有切分逻辑直接读 `content_list`(§3.1.2 切分主流程基于 mineru block 结构,不基于 markdown 字符流) |
| `content_list` | `content_list_v2.json` | 页级嵌套结构(详见 §2.4.4.1);chunking 阶段是**唯一输入**,用作:① 目录页提取本书目录权威清单(§3.1.2 Step 1)、② 正文 title block 匹配字典找节边界(§3.1.2 Step 2)、③ 节内 paragraph/title/list 等 block 累积切父块/子块、④ 识别表格/chart 块做双粒度处理、④ 噪音 type 过滤(黑名单);GIN 索引支持按 type 聚合查询。**注**:`title.level` 字段全是 1,无意义,不读 |
| `middle_data` | `middle.json` | 体积最大(典型 16-84MB,极端 300MB+),含 token 级 bbox,排查解析异常时使用 |
| `model_data` | `model.json` | 模型推理细节(典型 2-19MB),想看 mineru 模型对某块的 layout 分类置信度时翻它 |
| `pdf_path` | 文件系统 | 原始 PDF 路径引用,PDF 本体存本地磁盘 |

**不存入 `raw_documents` 表的内容**

- 原始 PDF 文件本体：体积大，存本地磁盘，表中只记路径
- `target_document_span.pdf` / `target_document_layout.pdf`：MinerU 调试用中间产物，不纳入系统存储

#### 2.4.4.1. `content_list_v2` 真实嵌套结构与 block.type 一览

mineru 2.x `content_list_v2.json` 的实测结构比早期 v1(扁平 list)复杂得多——**每个 block 的 `content` 不是字符串而是嵌套 dict,且不同 type 的内层 schema 各异**。下游(C1 mineru_loader / C2 chunking)在写代码消费此字段前必须按本表对照,否则会按 v1 的简化心智模型踩坑(典型错误:把 `block["content"]` 当 str 读、按 spec 早期描述的 `caption/body/footnote` 找表格字段)。

**顶层结构**:`list[页数] → list[block] → block = {"type": str, "content": dict, "bbox": [x0,y0,x1,y1]}`

**实测 block.type 分布**(以诊断学 第10版 626 页为参考样本):

| type | 数量 | 占比 | 是否进 chunks 表(§3.1.2) |
|---|---|---|---|
| `paragraph` | 4610 | 35% | ✓ 主体正文 |
| `title` | 2191 | 17% | ✓ 标题(level 重建见 §3.1.1 末) |
| `page_footer` | 1142 | 9% | ✗ 噪音 |
| `list` | 868 | 7% | ✓ 列表项 |
| `page_number` | 606 | 5% | ✗ 噪音 |
| `page_header` | 579 | 4% | ✗ 噪音 |
| `image` | 532 | 4% | **按 sub_type 分流**:`flowchart` 进 chunks 表(`chunk_type='figure'`,见 §3.1.2);`chemical / text_image / natural_image / None` 全丢(§3.1.1 末规则) |
| `table` | 177 | 1% | ✓ 进 chunks 表(`chunk_type='table'`,见 §3.1.2) |
| `chart` | 74 | <1% | ✓ 进 chunks 表(归入 `chunk_type='figure'`;mineru chart 识别质量差,统一按图片处理,见 §3.1.2) |
| `equation_interline` | 54 | <1% | ✗ 丢(数量小且公式通常已在所属段落文字描述里带过,§3.1.1 末规则) |

噪音 type(`page_header / page_footer / page_number / image content`)合计 ~22%,**chunking 阶段必须显式过滤,否则会把页眉页脚页码当正文切进 chunks** ——具体白名单与 extractor 规则见 §3.1.2。

**各 type 的 `content` 内层 schema**:

```python
# title
{"type": "title", "content": {"title_content": [{"type": "text", "content": "诊断学"}], "level": 1}, "bbox": [...]}
# paragraph
{"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "..."}]}, "bbox": [...]}
# list (深 4 层嵌套,可能含 ordered/unordered)
{"type": "list", "content": {"list_type": "text_list",
                              "list_items": [{"item_type": "text", "item_content": [{"type": "text", "content": "..."}]}]}, "bbox": [...]}
# table (caption/footnote 字段名带 table_ 前缀,正文是 HTML 字符串)
{"type": "table", "content": {"image_source": {"path": "images/xxx.jpg"},
                               "table_caption":  [{"type": "text", "content": "表1-1 ..."}],
                               "table_footnote": [],
                               "html": "<table><tr><td>...</td></tr></table>"}, "bbox": [...]}
# chart (含曲线图被 OCR 成的 markdown 数据表,字段名带 chart_ 前缀)
{"type": "chart", "content": {"image_source": {"path": "images/xxx.jpg"},
                               "chart_caption":  [{"type": "text", "content": "..."}],
                               "content": "| 列1 | 列2 |\n| --- | --- |\n| ... |"}, "bbox": [...]}
# image (content 字段 50% 含 VLM 幻觉,loader 必丢,见 §3.1.1 末)
{"type": "image", "content": {"image_source": {"path": "images/xxx.jpg"},
                               "image_caption":  [{"type": "text", "content": "图1-1 ..."}],
                               "image_footnote": [],
                               "content": "..."  # ← 必丢
                               }, "bbox": [...]}
# equation_interline (行间公式;mineru 也有 equation_inline 行内公式但本样本未出现)
{"type": "equation_interline", "content": {"math_content": "\\frac{a}{b}", "math_type": "latex",
                                            "image_source": {"path": "images/xxx.jpg"}}, "bbox": [...]}
# page_header / page_footer / page_number (噪音,直接丢)
{"type": "page_header",  "content": {"page_header_content":  [{"type": "text", "content": "+ "}]}, "bbox": [...]}
{"type": "page_footer",  "content": {"page_footer_content":  []}, "bbox": [...]}
{"type": "page_number",  "content": {"page_number_content":  []}, "bbox": [...]}
```

**与早期 spec 描述的勘误**(以下旧描述均已作废,以本表为准):
- ❌ "block.content 是字符串" → ✓ 是嵌套 dict,不同 type 内层 key 不同
- ❌ "table 字段名为 caption/body/footnote" → ✓ 实际为 `table_caption / html / table_footnote`,且 body 为 HTML 字符串(下游想要 row-level 数据需自行 parse HTML)
- ❌ "block.type 只有 title/paragraph/table/image/equation 5 种" → ✓ 实际 10 种(多 page_header/page_footer/page_number/list/chart)



### 2.4.5. 病人信息：PostgreSQL

> 表结构对齐八大采集规范（主诉→现病史→既往史→过敏史→用药史→个人史→婚育史→家族史），确保问诊采集到的每一项都有持久化落点。其中主诉和现病史由 `info_collect` ① 从 `patient_input` 实时提取（存 State RAM），其余六项作为患者历史档案从本库加载。

```
users (账号系统)
  └── patients (1:1，基本信息 + 个人史)
        ├── medical_history         (1:N，基础疾病 + 传染病 ⚠️必问)
        ├── surgical_trauma_history (1:N，手术与外伤 ⚠️必问)
        ├── transfusion_history     (1:N，输血史)
        ├── allergies               (1:N，过敏史 ⚠️安全底线)
        ├── medications              (1:N，用药史 ⚠️必问)
        ├── family_history          (1:N，家族史)
        ├── menstrual_reproductive  (1:1，女性婚育/月经史)
        └── exam_reports            (1:N，检查报告上传)
```

具体设计如下

```sql
-- 用户认证表
users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,         -- 存储哈希后的密码
  role VARCHAR(20) NOT NULL,      -- patient / doctor / admin 等
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)
```
```sql
-- 患者基本信息 + 个人史（关联 users 表）
-- 对应采集规范：基本信息 + （六）个人史
patients (
  id UUID PRIMARY KEY REFERENCES users(id),
  name TEXT,
  gender VARCHAR(10),             -- male / female / other
  birth_date DATE,
  blood_type VARCHAR(20),         -- 血型，如"AB-Rh(D)阴性"，急诊相关
  height_cm INT,
  weight_kg DECIMAL(5,1),
  phone TEXT,
  emergency_contact TEXT,         -- 紧急联系人姓名+电话
  -- 个人史字段（低基数，直接内嵌）
  smoking_status VARCHAR(20),     -- never / former / current
  smoking_pack_years DECIMAL(5,1),-- 包年数（每日包数×年数）
  alcohol_status VARCHAR(20),     -- never / occasional / regular / heavy
  alcohol_detail TEXT,            -- 频率、每日酒精摄入量
  occupation TEXT,                -- 职业
  occupational_exposure TEXT,     -- 粉尘、化学毒物、放射线、噪声等职业暴露
  travel_history TEXT,            -- 近期旅居史（疫区/特殊地区）
  infectious_contact TEXT,        -- 传染病接触史
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)
```
```sql
-- 既往病史：基础疾病 + 传染病（一对多） ⚠️必问
-- 对应采集规范：（三）既往史 - 基础疾病史⚠️必问 + 传染病史
medical_history (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  category VARCHAR(20) NOT NULL,  -- chronic（基础病）/ infectious（传染病）
  condition TEXT NOT NULL,        -- 疾病名称，如"2型糖尿病"、"乙型肝炎"
  icd10_code VARCHAR(10),        -- ICD-10 编码（可选，便于结构化检索）
  diagnosed_at DATE,
  resolved_at DATE,               -- NULL 表示持续中
  control_status VARCHAR(20),     -- well_controlled / poorly_controlled / unknown
  notes TEXT,                     -- 目前控制情况等补充说明
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)

-- 手术与外伤史（一对多） ⚠️必问
-- 对应采集规范：（三）既往史 - 手术与外伤史⚠️必问
surgical_trauma_history (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  type VARCHAR(10) NOT NULL,      -- surgery / trauma
  name TEXT NOT NULL,             -- 手术名称 或 外伤描述
  occurred_at DATE,
  hospital TEXT,                  -- 手术医院（可选）
  has_complications BOOLEAN DEFAULT FALSE,
  complications TEXT,             -- 并发症描述
  sequelae TEXT,                  -- 后遗症描述
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)

-- 输血史（一对多）
-- 对应采集规范：（三）既往史 - 输血史（传染病筛查及免疫反应风险评估）
transfusion_history (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  transfusion_date DATE,
  blood_product VARCHAR(30),      -- whole_blood / rbc / plasma / platelet 等
  reason TEXT,                    -- 输血原因
  adverse_reaction BOOLEAN DEFAULT FALSE,
  reaction_detail TEXT,           -- 不良反应描述
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)

-- 过敏史 ⚠️ 安全底线，必问
-- 对应采集规范：（四）过敏史
allergies (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  allergen TEXT NOT NULL,         -- 过敏原，如"青霉素"、"海鲜"、"花粉"
  allergen_type VARCHAR(20),      -- drug / food / environmental / material / other
  reaction TEXT,                  -- 过敏反应描述
  reaction_type VARCHAR(30),      -- rash / anaphylaxis / gi_reaction / angioedema 等
  severity VARCHAR(10),           -- mild / moderate / severe / life_threatening
  status VARCHAR(20) DEFAULT 'suspected', -- confirmed / suspected / resolved
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)

-- 用药史 ⚠️ 必问（含当前用药与历史用药）
-- 对应采集规范：（五）用药史
medications (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  drug_name TEXT NOT NULL,
  drug_category VARCHAR(30),      -- anticoagulant / hypoglycemic / hormone / immunosuppressant / otc / herbal / supplement 等
  dosage TEXT,                    -- "500mg"
  frequency TEXT,                 -- "每日两次"
  route VARCHAR(20),              -- oral / injection / topical 等
  started_at DATE,
  ended_at DATE,                  -- NULL 表示仍在服用
  prescribed_by TEXT,             -- 开药来源备注
  is_self_medication BOOLEAN DEFAULT FALSE, -- 自行购药 vs 处方
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)

-- 家族史（一对多）
-- 对应采集规范：（八）家族史
family_history (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  relation VARCHAR(20) NOT NULL,  -- father / mother / sibling / grandparent 等
  condition TEXT NOT NULL,        -- 疾病名称：遗传病、肿瘤、心脑血管、糖尿病、高血压、精神疾病等
  condition_category VARCHAR(30), -- genetic / cancer / cardiovascular / metabolic / psychiatric / other
  onset_age INT,                  -- 发病年龄（可选）
  notes TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)

-- 女性婚育/月经史（一对一）
-- 对应采集规范：（七）婚育史（女性必问）
menstrual_reproductive (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id) UNIQUE,
  menarche_age INT,               -- 初潮年龄
  cycle_days INT,                 -- 月经周期（天）
  period_days INT,                -- 经期天数
  last_menstrual_period DATE,     -- 末次月经（LMP）⚠️ 关键
  is_pregnant BOOLEAN,            -- 是否在孕
  gravidity INT,                  -- 孕次
  parity INT,                     -- 产次
  is_lactating BOOLEAN,           -- 是否在哺乳期（影响用药选择）
  menopause_age INT,              -- 绝经年龄（NULL 表示未绝经）
  notes TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)

-- 检查报告上传（一对多）
-- 对应 info_collect ① Step 3：加载患者已上传的检查报告
exam_reports (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  report_type VARCHAR(30) NOT NULL, -- blood_routine / urine_routine / biochemistry / imaging / ecg / physical_exam / pathology / other
  report_name TEXT,               -- 报告名称，如"2024年度体检报告"
  file_path TEXT,                 -- 上传文件存储路径（图片/PDF）
  file_mime VARCHAR(50),          -- image/jpeg / application/pdf 等
  report_date DATE,               -- 报告日期
  llm_summary TEXT,               -- LLM 阅读理解后的结构化摘要
  uploaded_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
)
```

### 2.4.6. 术语向量库：Milvus（terms_collection）

`terms_collection` 是独立于医学文献向量库(2.4.1)的专用术语检索库,**EL 移除后运行时不再被查询**(原服务的 ② build_query Entity Linking + 3.2.1 术语扩展两路均已下线,见 §4.1.6.2)。数据资产保留(40k+ ICD-10-CN alias 向量)作为未来重新启用 EL 或切换到 EL_DESIGN_REVIEW §11.6 方案 A/B 时的基础;`src/db/milvus/terms_collection.py` DAL 模块、`terms/build_icd10.py` 灌库脚本继续存在但运行时不 import。

**数据来源（三层叠加，优先级从高到低）**：

| 层级 | 来源 | 内容 | 获取方式 |
|------|------|------|---------|
| Layer 1 PROJECT | 项目自建口语词典 | 患者口语、俗称 → 标准术语映射（如"肚子疼"→腹痛） | 医师意见整理 + 上线后查询日志回流持续补充 |
| Layer 2 ICD-10-CN | 国家医保局临床版 | 中国医院实际使用的疾病编码，含中文标准名称和部分别名 | 国家医保局官网免费下载 |
| Layer 3 CMeSH | 中国医学主题词表 | 症状/解剖术语的中文规范名称与同义词，由中国医学科学院维护 | 官网免费申请 |

**核心设计原则**：一条记录对应一个别名（alias），多别名同属一个 concept_id，向量化 alias 文本而非 preferred_term，使口语/缩写/英文专业术语均可通过向量检索命中标准术语。

**Milvus Collection Schema（terms_collection）**：

```
{
    "id":             str,          # 记录唯一 ID：{concept_id}_{alias_index}
    "concept_id":     str,          # 概念唯一 ID：优先用 ICD-10-CN 编码（如 "R10.4"）；
                                    # 无 ICD-10-CN 编码时用 CMeSH ID；
                                    # 两者均无时用项目自赋 ID（PROJECT_{hash}）
    "preferred_term": str,          # 该概念的标准首选术语，如"腹痛"
    "alias":          str,          # 本条记录的别名文本，如"肚子疼"/"腹部疼痛"/"abdominal pain"
    "source_vocab":   str,          # 别名来源：PROJECT / ICD10CN / CMESH
    "icd10":          str,          # ICD-10-CN 编码，如 "R10.4"（无映射时为空）
    "category":       str,          # 概念类型：symptom / disease / anatomy / drug
    "dense_vector":   List[float]   # alias 文本的 Qwen3-Embedding-8B 向量，4096 维（仅 Dense，不需要 Sparse）
}
```

**与 2.4.1 的区别**：

| | 医学文献向量库（2.4.1） | 术语向量库（terms_collection） |
|---|---|---|
| 内容 | 医学指南/教材 Chunk | 术语别名条目 |
| 向量文本 | 原文/摘要/假设问题 | alias 字符串 |
| 检索目的 | 召回诊疗依据 | 实体归一化编码 |
| BM25 全文检索 | ✅ original 有（Milvus 内置 BM25） | ❌ 不需要 |
| 更新频率 | 随文档导入更新 | 随 ICD-10-CN/CMeSH 版本更新，PROJECT 层持续补充 |

**索引**：

```
# 向量索引（Dense 检索用）
collection.create_index(
    field_name="dense_vector",
    index_type="HNSW",           # 适合中等数据量、高召回场景
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)

# 标量索引（过滤 & 查询用）
collection.create_index(field_name="concept_id", index_type="INVERTED")   # 按 concept_id 查所有别名（用于术语扩展）
collection.create_index(field_name="category", index_type="INVERTED")     # 按类型过滤（仅查 symptom 等）
```
