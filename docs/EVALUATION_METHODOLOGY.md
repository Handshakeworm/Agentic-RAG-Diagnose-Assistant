# 评测方法学

> 本项目采用**三层独立评测**覆盖 RAG 检索 + LLM 诊断 + RAGAS 4 指标。
> 检索层完整对照实验(sparse 字段 / RRF 加权 / Reranker 全 K)详见 [RETRIEVAL_EVAL.md](RETRIEVAL_EVAL.md);本文档侧重**方法学** —— 评什么、怎么评、指标公式、设计取舍。

---

## 1. 总览

### 1.1 三层评测分工

| 层 | 评测目标 | Judge 输出 | 主指标 | 详细方法 |
|---|---|---|---|---|
| **检索层** | RAG 召回的教材章节是否覆盖正确知识点 | parent 级 **0~3 分** 作 ground truth | NDCG@20 / Hit@20 / MRR / Spearman ρ | 本文 §3 + [RETRIEVAL_EVAL.md](RETRIEVAL_EVAL.md) |
| **诊断层** | LLM 基于召回 chunks 给出的诊断是否等价 gold | 每 gold 一条 **6 档 match_type** | Top-1 / Top-2 / multi-gold recall | 本文 §4 |
| **RAGAS** | RAG 链路的 Faithfulness / Relevancy / Precision / Recall | 4 个独立 Pydantic schema + LLM Judge + embedding cosine | Faithfulness / Answer Relevancy / Context Precision / Context Recall | 本文 §5 |

三层各自独立 LLM Judge,结果**可交叉验证**(诊断层 top-1 高但 RAGAS Faithfulness 低 → LLM 答对但没引用召回 → 召回贡献度低;诊断层低但检索层 NDCG 高 → 召回好但 LLM 推理弱)。

### 1.2 通用约定

- **Judge 模型**:全程 **DeepSeek-v4-pro**(production 主链路同款,Judge 跟用户实际请求行为一致)
- **调用方式**:`llm.with_structured_output(Schema, method="json_mode").with_retry(stop_after_attempt=3)`(json_mode 兼容 DeepSeek reasoner;不用 json_schema/function_calling 避免 BadRequest)
- **并发**:`ThreadPoolExecutor`,case 间并发 case 内串行;case-level 文件 immediate persistence + `--resume` 断点续传
- **失败容忍**:单 Judge 失败该指标记 `None`/`-1`,不阻塞其他 Judge

---

## 2. 评测数据(62 case 执业医考题)

### 2.1 数据来源

62 道**中国执业医师考试**真题(教材原版 PDF → `.eval/rag_eval/parse_pdf_to_cases.py` 解析 → `.eval/rag_eval/cases/*.json`)。
每题包含完整的患者描述(主诉/现病史/体征/影像)+ 教科书"初步诊断"标准答案 + 诊断依据要点 + 鉴别诊断方向。

**不预筛**:62 case 全量入评测链路,不按科室 / KB 覆盖度 / 难度过滤 —— 系统对所有数据的真实表现就是评测目标。

### 2.2 Case JSON Schema

```json
{
  "case": "001_右额颞急性硬膜外血肿",
  "patient_text": "男性,23 岁,因骑车进行中被汽车撞倒...",   // 患者完整描述(query)
  "gold_diagnosis": ["右额颞急性硬膜外血肿"],                 // 主诊断 list(多 gold case 多项)
  "gold_evidence": [                                          // 教科书诊断依据要点(逐条)
    "有明确的外伤史",
    "有典型的中间清醒期",
    "头部受力点处有线形骨折",
    "出现进行性颅内压增高并脑疝"
  ],
  "gold_differential": ["急性硬膜下血肿:..."]                 // 教科书鉴别诊断
}
```

### 2.3 数据特点

- **单主诊断为主**:62 case 中约半数是单 gold(如"急性阑尾炎"),约半数是**多 gold case**(并存疾病/合并症,如 case 014 "脾破裂 + 肋骨骨折"、case 028 "Graves 病 + 甲亢性心脏病"、case 043 "冠心病急性心梗 + 急性左心衰竭")
- **覆盖系统多**:内科 / 外科 / 妇产 / 儿科 / 急诊均有,科室分布不强求均衡
- **真实文本噪声保留**:`patient_text` 是 PDF 解析原貌(如"头疼""烦燥""BP139-80"等口语化/格式不规范),不打磨 —— 同义词归一是 EL/Embedding 的责任,不是评测数据层

---

## 3. 检索层评测(RAG 召回质量)

### 3.1 评什么

RAG 召回的医学教材章节(parent)是否**覆盖该 case 诊断推理所需的知识点**。
不评 LLM 诊断准确率(那是 §4),只评检索本身好不好。

### 3.2 数据流

```
case.patient_text + state(全填充)
    │
    │ ① 跑 production RAG:Dense(Qwen3-Embedding) + Sparse(BM25 多字段 N 路)→ RRF 加权
    ▼
chunks_top200(顺序)
    │
    │ ② 顺序去重得 weighted_top50 parents(parent 不入向量,需 chunk → parent_id 映射)
    ▼
50 unique parents/case × 62 case = 3100 (case, parent) 对
    │
    │ ③ LLM Judge 每对评 0~3 分
    │    输入:patient_text + gold_diagnosis + parent.heading_path + parent.chunk_raw_text(median 1346 字)
    ▼
parent_scores → 算 NDCG / Hit@K / MRR / Spearman
```

### 3.3 LLM Judge 设计(0~3 评分)

prompt 给 LLM **诊断答案作背景**(让评判贴近临床相关性),输出严格 4 档:

| score | 标准 | 例子 |
|---|---|---|
| **3** 高度相关 | 直接讨论本病的诊断要点 / 鉴别 / 病理生理 / 治疗 | case 001 EDH ↔ "硬膜外血肿核心章节" |
| **2** 中度相关 | 同系统疾病、相似机制、合并诊断 | case 001 EDH ↔ "颅脑外伤总论" |
| **1** 弱相关 | 同医学领域但跟本病无关 | case 001 EDH ↔ "脑膜瘤" |
| **0** 完全无关 | 不同系统疾病、非疾病章节 | case 001 EDH ↔ "消化系统止吐药" |

**fusion-agnostic 设计**:score 通用,后续 RRF 权重 / Reranker 等任何 fusion 调参方案的 Top-K parents 跟当前 N/5 高度重叠,可复用本份 score 少量增量补评。

**4 档分布**(全 62 case):0(25.2%)/ 1(27.7%)/ 2(25.4%)/ 3(21.6%) —— Judge 区分度好,抽样准确(零误判)。

### 3.4 指标定义

| 指标 | 公式 | 含义 |
|---|---|---|
| **NDCG@K**(指数形式) | `Σ (2^rel - 1) / log₂(rank+1) / IDCG@K` | 排序质量(高分在前 = 高 NDCG)|
| **Hit@K(≥3)** | Top-K 内**至少 1 个** parent ≥ 3 分 | 漏召率 = 1 - Hit |
| **MRR(≥2)** | `1 / 第一个 score ≥ 2 chunk 的 rank` | 第一个相关 chunk 在多前 |
| **Precision@K(≥2 / ≥3)** | Top-K 中达阈值数 / K | 召回密度 |
| **Spearman ρ** | prediction 顺序 vs LLM ground truth 顺序的秩相关 | 排序对齐度 |
| **Avg score @K** | Top-K parents 平均分 | 整体相关性 |

**双粒度**(chunk-level + parent-level):chunk score 派生 = `parent_scores[parent_of(chunk)]`,**共用一份 Judge 数据,评测成本不增加**。parent-level 对齐 ⑩ diagnose LLM 实际输入,是主决策口径。

### 3.5 已跑结果(parent-level macro,全 62 case)

| K | NDCG | P(≥2) | P(≥3) | MRR(≥2) | Hit(≥3) | Avg |
|---|---|---|---|---|---|---|
| 5 | 0.793 | 83.9% | 62.6% | 0.952 | 96.8% | 2.448 |
| 10 | 0.757 | 77.1% | 50.5% | 0.954 | 100% | 2.226 |
| **★20**(生产) | **0.774** | **67.6%** | **37.8%** | **0.954** | **100%** | **1.944** |
| 50 | 0.892 | 47.0% | 21.6% | 0.954 | 100% | 1.435 |

**Spearman ρ(parent-level 全 50 vs LLM ground truth)**:mean **+0.708** / median +0.725 / std 0.106

**关键解读**:
- **MRR(≥2) = 0.954** — 几乎所有 case Top-1 即命中中度以上相关,**顶部不是问题**
- **Hit@20(≥3) = 100%** — **没有 case 漏掉高相关章节**(0 漏召)
- **ρ = +0.71** — RAG 排序高度对齐 LLM 判断,改进空间有限
- chunk-level 数值高于 parent-level:同 parent 的多 child 重复占位虚高 → **parent-level 更代表真实业务价值**

→ **完整 7 章对照实验**(sparse 字段 / RRF 等权 vs 加权 / Reranker 全 K 性价比 / 表格召回 Q1/Q2/Q3)见 [RETRIEVAL_EVAL.md](RETRIEVAL_EVAL.md)

---

## 4. 诊断层评测(LLM 诊断准确率)

### 4.1 评什么

LLM ⑩ diagnose 节点的 candidates list **是否覆盖 gold 标准诊断**,以"临床本质等价"判定。
不评检索好不好(那是 §3),只评最终诊断输出。

### 4.2 数据流

```
case.patient_text + RAG 召回的 Top-20 parents + figure 截图
    │
    │ ① 跑 production ⑩ diagnose:vision LLM(qwen3.5-plus)
    ▼
candidates: [{disease, probability, evidence, differentiation}, ...]
    │
    │ ② LLM Judge 评 gold ↔ LLM 候选等价性
    │    关键:**只给 Judge 看 gold + candidates 疾病名**,不给 patient_text
    │           — 防止 Judge 自己重新做诊断推理,只评"名词等价性"
    ▼
per-gold verdict: {gold_item, matched_rank, match_type, reason}
    ▼
aggregate → top1/top3/top5 + multi-gold recall
```

### 4.3 LLM Judge 设计(6 档 match_type)

按"**临床本质等价**"评判,不按字面纠结:

| match_type | 含义 | 例子 |
|---|---|---|
| **exact** | 字面完全相同 | gold="急性阑尾炎" ↔ LLM="急性阑尾炎" |
| **equivalent** | 临床本质同一疾病,只是表述差异(同义词 / 部位细微差 / 中英文混排) | gold="Graves 病" ↔ LLM="毒性弥漫性甲状腺肿";gold="右额颞 EDH" ↔ LLM="右颞部硬膜外血肿" |
| **more_specific** | LLM 比 gold 更精确分型(**算等价命中** —— 多给信息是好事) | gold="急性心梗" ↔ LLM="ST 段抬高型急性心梗" |
| **more_general** | LLM 比 gold 笼统 | gold="左肾结核" ↔ LLM="泌尿生殖系结核" |
| **partial** | 多 gold 中仅命中部分(合并诊断 gold="A+B" 但 LLM 只列 A) | gold=["脾破裂","肋骨骨折"] ↔ LLM 只列了"脾破裂" |
| **none** | 完全不同病 | gold="输尿管结石" ↔ LLM="输尿管肿瘤" |

### 4.4 指标定义

**主诊断指标**(以 `gold[0]` 为评判对象):

| 指标 | 公式 | 用途 |
|---|---|---|
| `top1_hit_loose` | gold[0] 在 LLM rank=1 且 match_type ∈ {exact, equivalent, more_specific, more_general, partial} | 宽口径 — "LLM 第一选项就是 gold 临床等价"(患者视角:大方向对就能引导分诊)|
| `top1_hit_strict` | gold[0] 在 LLM rank=1 且 match_type ∈ {exact, equivalent, more_specific} | 严格口径 — 排除 `more_general`/`partial`(医生视角:精确分型重要)|
| `topK_hit` | gold[0] 在 LLM rank ≤ K(loose 口径) | K=3 / K=5 |

**多 gold 指标**(衡量合并症 / 并存疾病覆盖):

| 指标 | 公式 | 用途 |
|---|---|---|
| `multi_gold_recall.mean` | 各 case 命中 gold 数 / 总 gold 数,跨 case macro-average | 多并发诊断 case 平均覆盖率 |
| `n_full_recall_cases` | 所有 gold 都被 LLM 命中的 case 数 | 全召回 case 数 |

### 4.5 已跑结果

| 主指标 | 数值 | 解读 |
|---|---|---|
| **Top-1 命中率(loose)** | **93.5%** (58/62) | LLM 第一选项就是 gold 临床等价 |
| Top-1 命中率(strict) | 61.3% (38/62) | 严格口径 |
| **Top-2 命中率** | **100%** (62/62) | gold 主诊断全部在前 2 —— 4 个非 top-1 命中 case 全部排 rank=2 |
| Top-5 命中率 | 100% | |
| **Multi-gold recall mean** | **0.867** | 多并发诊断 case 平均 86.7% gold 被覆盖 |
| 全 gold 命中 case | 45/62 (72.6%) | |
| **0 主诊断方向错** | **0/62** | 没有任何 case 主诊断 match_type=none |

**match_type 分布**(62 主诊断):exact 9 / equivalent 17 / more_specific 15 / more_general 13 / partial 8 / none 0

数据原文:[`.eval/rag_eval/diagnose_judge_summary.json`](../.eval/rag_eval/diagnose_judge_summary.json)

### 4.6 3 个设计取舍 Q&A

**Q1:为什么 Judge 不看 `patient_text`?**
让 Judge 看 patient_text 会让它**自己重新做诊断推理**(看完患者描述后倾向认可一些 candidate),偏离"名词等价性"评判。只给 gold + candidates 强制 Judge 走纯术语对照路径。

**Q2:为什么 loose 和 strict 两口径?**
`more_general`(LLM 比 gold 笼统)在临床上是**部分命中**:LLM 给"颅内血肿"vs gold"硬膜外血肿"方向对但失精确度。loose 算命中(站患者角度大方向对就能引导分诊),strict 不算(站医生角度精确分型重要)。两个口径都报让读者按场景取舍。

**Q3:为什么按 match_type 分布而不是单一分数?**
单一"准确率%"丢失大量信息。`equivalent`/`more_specific` 多 → LLM 表达力强 / 有自主分型能力;`partial` 多 → 多 gold 召回弱;`none` 多 → 真有跑偏。这些信号比单一数字更能指导后续优化。

---

## 5. RAGAS 4 指标评测

### 5.1 评什么 + 为什么不用 `ragas` 库

[ragas](https://github.com/explodinggradients/ragas) 是 RAG 评测事实标准,本项目**对齐 RAGAS 算法 + 自实现 LLM Judge**,不用 ragas 库,因 3 个限制:

1. **默认 OpenAI 调用**:ragas 内部 `LangchainLLMWrapper` 默认走 OpenAI,改 DeepSeek 需要适配 LangChain LLM 接口(可行但工作量),且 ragas v0.2 部分指标 prompt 强依赖 OpenAI 风格输出
2. **prompt 全英文**:ragas 内置 Faithfulness / Answer Relevancy prompt 是英文,中文医疗术语(如"硬膜外血肿"、"Graves 病")命中率低,需要重写 prompt
3. **embedding 依赖**:Answer Relevancy 默认 `text-embedding-ada-002`,本项目已有 Qwen3-Embedding-8B(4096 维,中文医疗专域),换 embedding 也得改 wrapper

**自实现代价**:多写 ~400 行代码([.eval/rag_eval/run_ragas_judge.py](../.eval/rag_eval/run_ragas_judge.py))。
**收益**:跟 production 完全同款 LLM + Embedding,prompt 适配中文医疗,**算法严格对齐 ragas v0.2**。

### 5.2 数据流

```
case (gold_diagnosis + gold_evidence + patient_text + candidates + chunks_top200_scanned)
    │
    │ ① 提取 top-K parents(K=20,对齐 ⑩ diagnose 真实输入)
    │    .eval/rag_eval/run_ragas_judge.py::take_top_parents
    ▼
top-20 parent chunks (raw_text + heading_path,批查 PG chunks 表)
    │
    │ ② 4 个 RAGAS 指标 per case 各跑 1 次 LLM(+ Answer Relevancy 加 1 次 embedding)
    ▼
    ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
Faithfulness        Answer Relevancy      Context Precision     Context Recall
拆 LLM evidence       LLM 反推 N=3 q       每 parent 0/1 verdict   gold_evidence 逐条
为 atomic claims      → Qwen3-Embedding    → rank-aware MAP@K      判 supported
逐条判 supported       cosine 均值                                 (gold ground truth)
    │                    │                       │                       │
    ▼                    ▼                       ▼                       ▼
ragas_judge/<case>.json(per case 4 指标 + claim/q/chunk 级 details)
    ▼
aggregate → ragas_summary.json (macro-average + min/max/失败率)
```

### 5.3 四指标算法详解

#### 5.3.1 Faithfulness — LLM 答案对 retrieved context 的忠实度

| | |
|---|---|
| **衡量什么** | LLM 诊断 evidence 里每条陈述是否能在 retrieved 教材片段里找到支持(防幻觉)|
| **RAGAS 原算法** | 2 步 LLM:① `StatementsSimplificationPrompt` 拆 atomic statements;② `NLIStatementsPrompt` 对每条做 NLI(entailment 判定)|
| **本实现** | **1 步合并**(成本减半):LLM 同时拆 claims + 判 supported,prompt 强调"NLI entailment 不是 supported" |
| **score 公式** | `supported_claims / total_claims` |
| **关键 prompt 约束** | 不接受 LLM 在原文加的"(文本块 X)"引用作为依据 — 必须 retrieved 片段原文真的有相关内容 |
| **简化代价** | 对齐度 ~85%(RAGAS 2 步 ~100%)。同次 LLM 拆+判可能让 claims 粒度偏粗(倾向于拆少不拆多),supported 率可能系统性偏高 5-10pp |
| **关键代码** | `judge_faithfulness` + `_FAITHFULNESS_PROMPT` |

#### 5.3.2 Answer Relevancy — LLM 答案对 query 的切题度

| | |
|---|---|
| **衡量什么** | LLM 诊断答案是否针对该患者描述(不评诊断对错,只评是否在回答该患者)|
| **RAGAS 原算法** | LLM 反推 N=3 个 question + embedding cosine(本实现严格对齐)|
| **数据流** | ① LLM 看 answer 反推 3 个候选 question + `noncommittal` flag(LLM 是否含糊不答);② Qwen3-Embedding-8B 编码 `[patient_text, gen_q_1, gen_q_2, gen_q_3]`;③ 算 cosine 相似度 |
| **score 公式** | `mean(cos_sim(patient_text, gen_q_i)) × (1 - noncommittal)` |
| **为什么不 LLM 直评** | LLM 直接打 0-1 分主观 bias 大,不同 case 评分漂移。RAGAS 用"反推 q + cosine"是**为了量化**(cosine 是确定性数值,可比性强)|
| **关键代码** | `judge_relevancy_ragas` + `_REVERSE_Q_PROMPT` + `_cosine_sim_batch` |

#### 5.3.3 Context Precision — retrieved chunks 中相关比例(rank-aware)

| | |
|---|---|
| **衡量什么** | RAG 召回的 top-K parents 里有多少跟 gold diagnosis 相关,且**前面相关多 = 好**(rank 越前权重越大)|
| **RAGAS 原算法** | Rank-aware MAP@K(本实现严格对齐 ragas v0.2)|
| **数据流** | 对每个 parent LLM 判 `relevant ∈ {true, false}`(只看 gold 不看 LLM,排除 bias)|
| **score 公式** | `Σ_{i=1..K} (Precision@i × v_i) / Σ_{i=1..K} v_i`<br/>其中 `Precision@i = 前 i 个中 relevant 比例`,`v_i ∈ {0,1}` 是 chunk i 的 verdict |
| **vs binary precision** | binary 对 rank 不敏感(`[1,1,1,0,0]` 和 `[0,0,1,1,1]` 同分);rank-aware MAP 前者 1.0 后者 ~0.16,正确反映"想要的在前面" |
| **对照输出** | 同时报 `context_precision`(rank-aware,主指标)+ `context_precision_binary`(对照),让读者看差距 |
| **关键代码** | `judge_precision_ragas` + `_PRECISION_PROMPT` |

#### 5.3.4 Context Recall — gold 知识被 retrieved 覆盖比例

| | |
|---|---|
| **衡量什么** | gold 标准诊断的教科书要点(`gold_evidence`)有多少能在 retrieved chunks 里找到。缺失即 RAG 漏召信号 |
| **RAGAS 原算法** | 从 `ground_truth answer` 拆 statements,逐条判 attributable to contexts |
| **本实现** | 直接用 case JSON 已有的 `gold_evidence`(教科书要点 list,医生标注),逐条判 supported |
| **score 公式** | `supported_claims / total_claims` |
| **为什么不让 LLM 拆 statements** | `gold_evidence` 已经是医生预先标注的 ground truth 要点,直接用比 LLM 拆 ground_truth_answer 噪音小 —— 本质是**数据更优**版本(对齐度 90%+,RAGAS 等价但避免 LLM 拆错)|
| **关键代码** | `judge_recall` + `_RECALL_PROMPT` |

### 5.4 计算成本

| 项目 | 单 case 成本 | 62 case 全量 |
|---|---|---|
| LLM 调用(DeepSeek-v4-pro) | 4 次(4 指标各 1) | **248 次** |
| Embedding 调用(Qwen3-Embedding-8B) | 1 次(batch 4 段) | **62 次** |
| 单 case 耗时 | ~30-60s | ~10-15 min(workers=5)|
| API 成本(DeepSeek 报价) | ~¥0.02-0.05 | **~¥1-3** |
| GPU 显存(Qwen3-Embedding-8B INT8)| 9GB | 同上(singleton)|

### 5.5 跟 ragas 库结果可比性

跟 ragas v0.2 跑 OpenAI gpt-4o + text-embedding-ada-002 的结果**不能精确对比**:

- **LLM 不同**:DeepSeek-v4-pro vs gpt-4o,推理风格 / 严格度不同
- **Embedding 不同**:Qwen3-Embedding-8B(4096 维,中文医疗专域)vs ada-002(1536 维,通用)
- **Prompt 不同**:本实现中文医疗适配 vs ragas 内置英文通用

但**算法一致,指标语义可比**。本实现的数字反映"用 production 同款 LLM + 中文医疗 prompt 跑出的 RAGAS 风格指标",对项目自身的纵向对比(改 prompt / 换 Reranker / 调 RRF 权重前后)更有指导意义。

---

## 6. 评测产物 + 复现命令

### 6.1 文件结构

```
.eval/rag_eval/
├── cases/                                    # 62 case 输入(patient_text + gold)
│   └── 001_xxx.json ...
│
├── sparse_fusion_compare/                    # 检索层评测(§3)
│   ├── compare_result.json                   # 等权 vs 加权 RRF 全 chunk 落盘
│   └── judge_per_case/                       # 检索 LLM Judge 0~3 评分
│       ├── 001_xxx.json
│       ├── _meta.json
│       └── _parents_meta.json
│
├── diagnose_eval/                            # 诊断中间产物(retrieve + diagnose 输出,§4 输入)
│   └── 001_xxx.json ...                      # patient_text + candidates + chunks_top200_scanned
│
├── diagnose_judge/                           # 诊断 LLM Judge 结果(§4)
│   └── 001_xxx.json ...
├── diagnose_judge_summary.json               # 诊断层 aggregate
│
├── ragas_judge/                              # RAGAS 4 指标 LLM Judge(§5)
│   └── 001_xxx.json ...
└── ragas_summary.json                        # RAGAS aggregate
```

### 6.2 评测脚本入口

| 脚本 | 用途 | 对应方法 |
|---|---|---|
| `run_diagnose_eval.py` | 跑 production pipeline 出 diagnose_eval/*.json | §4 输入 |
| `run_llm_judge.py` | 检索层 LLM Judge 评 0-3 分 | §3 |
| `compute_metrics.py` | 检索层 NDCG / Hit / MRR 计算 | §3 |
| `compare_rrf_weighting.py` | RRF 等权 vs 加权对照 | §3 / RETRIEVAL_EVAL.md §4 |
| `compare_rerank_full.py` | Reranker 开关 K 维度全对照 | §3 / RETRIEVAL_EVAL.md §7 |
| `run_diagnose_judge.py` | 诊断层 LLM Judge 评等价性 | §4 |
| `run_ragas_judge.py` | **RAGAS 4 指标 LLM Judge** | §5 |

### 6.3 复现命令

```bash
# 准备:venv 激活 + .env 已配置(EMBEDDING_MODEL_PATH / LLM_API_KEY 等)
source .venv/bin/activate

# § 3 检索层(分两步:跑 retrieve+fusion → LLM Judge 评 0-3 → 算指标)
python .eval/rag_eval/compare_rrf_weighting.py
python .eval/rag_eval/run_llm_judge.py --workers 10
python .eval/rag_eval/compute_metrics.py

# § 4 诊断层(跑 production pipeline → LLM Judge 评等价性)
python .eval/rag_eval/run_diagnose_eval.py
python .eval/rag_eval/run_diagnose_judge.py --workers 10

# § 5 RAGAS(消费 § 4 产物,跑 4 指标 Judge)
# 前提:GPU 至少 10GB 可用显存(Qwen3-Embedding-8B 9GB);api 容器在跑要先 stop
docker compose stop api
python .eval/rag_eval/run_ragas_judge.py --workers 5

# 中途挂了 — 所有评测脚本都支持 --resume 断点续传
python .eval/rag_eval/run_ragas_judge.py --workers 5 --resume
```
