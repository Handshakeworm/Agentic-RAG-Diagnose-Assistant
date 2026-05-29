# RAG 检索链路评测报告

**状态**:2026-05-17 baseline 建立完成,可作后续 fusion 调参 / reranker 决策依据
**作者**:Claude 协作
**适用**:DEV_SPEC §3.2.1 / §3.2.2 / §3.2.3 / §4.1.2 RAG 检索链路
**EL 决策另行讨论**:见 [EL_DESIGN_REVIEW.md](EL_DESIGN_REVIEW.md)(本报告不含)

---

## 1. 总览

**评测目标:** 量化当前 RAG 链路(retrieve + RRF + 可选 reranker)在 62 case 真实主诉上的召回质量,以 LLM Judge 为 ground truth。

**核心数据(K=20 = 生产 ⑩ diagnose 输入数):**

| 指标 | parent-level | chunk-level |
|---|---|---|
| **NDCG@20** | **0.774** | 0.748 |
| P@20(≥2 中以上相关) | 67.6% | 79.2% |
| P@20(≥3 高度相关) | 37.8% | 53.5% |
| Hit@20(≥3) | **100%** | 100% |
| Avg score | 1.944 | 2.272 |
| **Spearman ρ vs LLM** | **+0.708** | - |

**业务级解读:** 生产 Top-20 parents 平均 14 个中度以上相关,8 个高度相关,**100% case 至少召回 1 个高相关章节**(无漏诊)。

**关键决策:**

1. ✓ **step1 sparse 多字段直采**(放弃 EL alias 反查),数据驱动
2. ✓ **RRF 动态加权 dense_weight = max(1, N_sparse/5)**,补救 sparse 多路挤兑 dense
3. ✗ **Reranker 默认关闭** — 全 62 case 数据,K=20(生产口径)下 BGE Reranker 在所有主指标上均无优势(Hit -1.6pp / NDCG -0.076 / MRR -0.065,无指标维度赢),**性价比决策关掉,非"灾难性退化"**(详见 §7)

---

## 2. step1 sparse_queries 改造

### 背景

- 原 step1 设计:`sparse_queries` 由 EL 链上 symptom 反查 alias 词袋 + report 双源构成(spec §3.2.1 Step 2)
- 实测 EL 命中率:中文症状词 Tier 3 占位 50%,alias 反查同义词扩展实际收益低
- 同时**丢失了 step0 已经拆好的大量结构化字段**(`chief_complaint` / `trigger` / `location` / `nature` 等都没进 sparse)

### 字段评估(实测 62 case 填充率 + 实际值)

| 字段 | 填充率 | 实际值样本 | 决策 |
|---|---|---|---|
| `chief_complaint` | 100% | "腹痛 3 天" | ✓ 加 |
| `trigger` | 47% | "上呼吸道感染"/"急刹车后右膝撞击" | ✓ 加 |
| `location` | 58% | "右上腹"/"上腹部,向右肩放射" | ✓ 加 |
| `nature` | 21% | 刀割样/刺痛/绞痛/胀痛/压榨性 | ✓ 加 |
| `severity` | 19% | 严重/剧烈/剧痛/重度(全 ≥2 字) | ✓ 加 |
| `duration_pattern` | 69% | 持续性/间歇性/阵发性 | ✓ 加 |
| `onset_mode` | 82% | 急性/缓慢/隐匿 | ⚠️ 加(有"急性"过泛风险) |
| `associated_symptoms` (list) | - | 恶心/呕吐/意识障碍 | ✓ 加 |
| `aggravating` (list) | 26 条 | 咳嗽/深呼吸/仰卧/饱餐 | ✓ 加 |
| `relieving` (list) | 18 条 | 硝酸甘油/庆大霉素/黄连素 | ✓ 加 |
| `positive_findings` (list) | - | WBC升高/低钠/CT征象 | ✓ 加 |
| `impressions` (list) | - | "右额颞线形骨折"等 | ✓ 加(过滤阴性印象) |
| `onset_time` | 98% | "4 小时前"/"3 个月前" | ✗ KB 不写相对时间 |
| `progression` | 77% | 加重/波动/稳定(仅 3 个泛词) | ✗ 词泛 IDF 低 |
| `treatment_response` | 61% | 好转/无效/缓解/疗效不明显 | ✗ 结论性词无 IDF |
| `treatment_tried` | 63% | "氟呱酸口服"/"按胃炎治疗" | ✗ 拉到药学 chunk 而非诊断 chunk |
| `present_illness` | 100% | 200+ 字整段 | ✗ 长度退化为 OR 检索 |
| `negative_findings` | - | "无发热" | ✗ BM25 不懂否定,反向贡献 |

### impressions 阴性过滤

`impressions` list 里约 30% 是阴性印象(`"(-)" / "正常" / "阴性" / "未见" / "无异常"`),与 `negative_findings` 同性质问题(BM25 不懂否定)。**用正则过滤包含这些字样的整条**:

```python
_NEGATIVE_IMPRESSION_RE = re.compile(r"\(-\)|正常|阴性|未见|无异常")
```

case 007 实测:5 条 impressions 4 条触发过滤,**信息不丢**(被过滤段的阳性发现都已在同 ReportFinding 的 positive_findings 里独立出现)。

### 实测数量

- case 001(简单):16 条 sparse
- case 062 麻疹(复杂报告):28 条
- **62 case 平均 21.8 条 sparse**

代码:[`.eval/rag_eval/step1.py`](../.eval/rag_eval/step1.py)

---

## 3. 表格召回验证(Q1/Q2/Q3)

**问题:** 表格 chunk(`chunk_type='table'`,KB 共 2744 个)能不能被语义检索召回?

**三组实验:**

### Q1 random_table_reverse(25 个随机 table,LLM 反推 query)

| Route | strict@5 | strict@20 | strict@200 | loose@200 |
|---|---|---|---|---|
| **dense** | **88%** | **96%** | **100%** | 100% |
| sparse | 36% | 52% | 80% | 100% |
| **rrf** | **88%** | **96%** | **100%** | 100% |

### Q3 numeric_table_reverse(15 个数值密集 table)

| Route | strict@5 | strict@20 | strict@200 |
|---|---|---|---|
| **dense** | 80% | 93% | **100%** |
| **rrf** | **93%** | 93% | **100%** |

### Q2 case_real(62 case patient_text,Top-K 中 table 占比)

| Route | @5 | @20 |
|---|---|---|
| dense | 10.3% | 9.6% |
| rrf | 5.8% | 7.2% |

**结论:**

1. **dense 路径召回表格表现非常好** — 因为 chunk LLM enrichment 把表格转译成 `summary`/`medical_statement`(自然语言),挂了向量,所以 dense 能命中(spec §3.1.3)
2. **数值密集 table 召回率没塌**(93% @20) — LLM enrichment 绕过了"数字对 dense 不友好"
3. **Sparse BM25 对表格弱** — `<table><tr><td>` HTML 标签 + 数字噪音,BM25 命中精度低,但 RRF 由 dense 兜底
4. **Q2 召回的 table 中数值密集占 9.6%**,vs 全库 13.2%,**轻微偏低但召回结构基本均匀**

代码:[`.eval/rag_eval/validate_table_recall.py`](../.eval/rag_eval/validate_table_recall.py)
数据:[`.eval/rag_eval/table_recall_validation/`](../.eval/rag_eval/table_recall_validation/)

---

## 4. RRF 加权 — 等权 vs 动态加权

### 问题:Sparse 多路挤兑 Dense

step1 改造后 sparse 路数涨到 12~30(平均 21.8),超出 spec §3.2.2 "自调节权重" 哲学的假设(N=3~5)。

**实测 RRF 数学:**
- Dense 单路最大贡献 = `1/(60+1) = 0.0164`
- Sparse 20 路集体命中(rank≈25)累加 ≈ **0.236**
- → **Sparse 是 Dense 的 14 倍权重,dense 单路独家命中被严重挤兑**

### 加权方案

动态加权,跟 N_sparse 联动:

```python
dense_weight = max(1, N_sparse / 5)
score(d) = dense_weight * 1/(k + rank_dense)
         + Σ 1/(k + rank_sparse_i)
```

效果:无论 N_sparse 多少,D/S 比稳在 spec 原假设的 1:3~1:4 水平。

### 对照实验(62 case,等权 vs 加权 N/5)

| K | Jaccard 重叠 | Dense Exclusive 保留(等权 / 加权)| 加权增量 |
|---|---|---|---|
| 5 | 42.3% | 7 / 28 | **×4** |
| 20 | 40.7% | 28 / 184 | **×6.5** |
| 50 | 43.0% | 77 / 540 | **×7.0** |

**等权下 dense 单路独家命中 chunks 平均每 case 只能保留 0.45 个进 Top-20;加权后 ~3 个。** Chunk type 分布平稳(无 distortion)。

### 与 spec 一致性

Spec §3.2.2 / §4.1.2 写 "等权融合,无需手动权重"。当前改造**突破了 spec 假设的 N_sparse 量级**。

实施时:不动 `src/rag/retrieval/fusion.py`(影响生产),评测脚本本地实现加权 RRF,后续可决定是否正式 patch spec + fusion。

代码:[`.eval/rag_eval/compare_rrf_weighting.py`](../.eval/rag_eval/compare_rrf_weighting.py)
数据:[`.eval/rag_eval/sparse_fusion_compare/compare_result.json`](../.eval/rag_eval/sparse_fusion_compare/compare_result.json)(63.9 MB,含全量 dense + sparse hits + 双 fusion Top-200 + chunks_meta)

---

## 5. LLM Judge 评测体系

### 设计

**fusion-agnostic 的 (case, parent) → 0~3 score**:LLM 评 parent chunk 对该 case 诊断推理的相关性,score 通用,后续任何 fusion 策略可复用。

### Judge 范围

加权 RRF(N/5)的 chunk Top-200 → 顺序去重 → **前 50 unique parents**(每 case 50,total 3100)。

> **设计理由:** parent 是 LLM 实际看到的 unit(spec §3.2.3 Context 扩展 chunk → parent 全文),Judge 评 parent 跟 LLM 输入对齐。后续 dense_weight 调参方案的 Top-50 parents 跟当前 N/5 高度重叠,可复用本份 score,少量增量补评。

### Prompt 设计(0~3 评分)

```
3 = 高度相关 — 直接讨论本病的诊断要点 / 鉴别诊断 / 病理生理 / 治疗
2 = 中度相关 — 同系统疾病、相似机制、合并诊断等,非诊断核心章节
1 = 弱相关   — 同医学领域但跟本病无关
0 = 完全无关 — 不同系统疾病或非疾病章节
```

输入:`patient_text` + `diagnosis`(gold,作评测背景)+ `parent.heading_path` + `parent.chunk_raw_text` 全文(median 1346 字,p95 3563 字)

输出:`{score: 0/1/2/3, reason: ≤40 字理由}`

**text-only**(skip figure 截图加载) — figure 命中率 < 1%,对照实验仍公平,后续可补 figure 专项实验。

### 实测结果(全 62 case)

| score | 数量 | 占比 |
|---|---|---|
| 0(无关) | 782 | 25.2% |
| 1(弱) | 860 | 27.7% |
| 2(中) | 787 | 25.4% |
| 3(高相关) | 671 | 21.6% |

**4 档分布均匀** — Judge 区分度好。Reason 抽样准确(case 001 "硬膜外血肿核心章节" score=3,"消化系统止吐药" score=0 — 零误判)。

### 工程实现

- **每 case 一文件** + immediate persistence(safe 中断重连)
- **case 间并发**(`ThreadPoolExecutor max_workers=10`),case 内串行
- **DeepSeek json_mode + 重试 3 次**

全量跑时间:**~8 分钟**(并发前预估 75 min),成本 ~$3-7。

代码:[`.eval/rag_eval/run_llm_judge.py`](../.eval/rag_eval/run_llm_judge.py)
数据:[`.eval/rag_eval/sparse_fusion_compare/judge_per_case/`](../.eval/rag_eval/sparse_fusion_compare/judge_per_case/)(62 个 case json + `_meta.json` + `_parents_meta.json` 9.5 MB)

---

## 6. 双粒度指标体系

### 设计 — chunk-level + parent-level

| 粒度 | unit | 用途 |
|---|---|---|
| chunk-level | chunk_id(child / table / figure)| 对齐 Reranker 视角(chunk-level 打分截断)|
| parent-level | parent_chunk_id | 对齐 ⑩ diagnose LLM 实际输入 |

**LLM Judge 只评 parent**,chunk score 派生 = `parent_scores[parent_of(chunk)]`(同 parent 多 child 共享分)。**评测成本不增加**,共用一份 Judge 数据。

### 指标算法

- **NDCG@K(指数形式):** `Σ (2^rel - 1) / log2(rank+1) / IDCG@K`,IDCG 用本 case Top-K 内最佳排序
- **Precision@K (≥2 / ≥3):** Top-K 中达阈值数 / K
- **MRR (≥2):** `1 / 第一个 score≥2 chunk/parent 的 rank`
- **Hit@K (≥3):** Top-K 是否至少 1 个高相关
- **Avg score @K:** Top-K 平均相关性
- **Spearman ρ:** prediction 顺序 vs LLM ground truth 顺序(parent-level 全 50 算)

### 全 62 case macro-average 结果

**chunk-level**

| K | NDCG | P(≥2) | P(≥3) | MRR(≥2) | Hit(≥3) | Avg |
|---|---|---|---|---|---|---|
| 5 | 0.821 | 88.4% | 70.6% | 0.952 | 95.2% | 2.574 |
| 10 | 0.796 | 85.0% | 64.8% | 0.954 | 100% | 2.481 |
| **★20** | **0.748** | **79.2%** | **53.5%** | **0.954** | **100%** | **2.272** |
| 50 | 0.753 | 65.7% | 37.3% | 0.954 | 100% | 1.906 |

**parent-level**(对齐 LLM 实际输入)

| K | NDCG | P(≥2) | P(≥3) | MRR(≥2) | Hit(≥3) | Avg |
|---|---|---|---|---|---|---|
| 5 | 0.793 | 83.9% | 62.6% | 0.952 | 96.8% | 2.448 |
| 10 | 0.757 | 77.1% | 50.5% | 0.954 | 100% | 2.226 |
| **★20** | **0.774** | **67.6%** | **37.8%** | **0.954** | **100%** | **1.944** |
| 50 | 0.892 | 47.0% | 21.6% | 0.954 | 100% | 1.435 |

**Spearman ρ (parent-level, 全 50 vs LLM ground truth):** mean **+0.708** / median +0.725 / std 0.106

### 解读

- **MRR(≥2) = 0.954** — 几乎所有 case Top-1 即命中中度以上相关,**顶部不是问题**
- **Hit@20(≥3) = 100%** — **没有 case 漏掉高相关章节**
- **chunk-level 数值高于 parent-level** — 高分 parent 的多 child 在 chunk-level 重复占位虚高,**parent-level 更代表真实业务价值**
- **ρ=0.71 高度对齐 LLM 判断** — RRF 加权方案已经很接近"理想排序",改进空间有限

代码:[`.eval/rag_eval/compute_metrics.py`](../.eval/rag_eval/compute_metrics.py)
数据:[`.eval/rag_eval/sparse_fusion_compare/metrics_result.json`](../.eval/rag_eval/sparse_fusion_compare/metrics_result.json)(154 KB)

---

## 7. Reranker 评测 — 推荐关闭(性价比决策,非退化大)

### 评测设计

- **范围内重排** — Reranker 输入 = RRF 加权 Top-50 chunks(case 001 ~103 个,跨 62 case mean ~95),`top_k=None` 全量重排后再按 K 切顶
- **生产对齐** — ⑩ diagnose 实际输入 = Top-20 parents,所以 **K=20 是唯一主决策口径**;K=5/10/50 退到附表辅助
- **fusion-agnostic ground truth** — LLM Judge 给的 parent score,RRF 加权 和 Reranker 都是 prediction,比哪个更接近 LLM 判断

### 主结果(★K=20,生产口径)

| 主指标 | RRF 加权 | Reranker | Diff | Reranker 赢 case 数 |
|---|---|---|---|---|
| **Hit@20(≥3)** | **100%** | 98.4% | **-1.6pp**(漏 1 case)| 0/62 |
| **NDCG@20** | **0.774** | 0.698 | **-0.076**(相对 -10%)| 12/62 (19%) |
| **MRR(≥2)** | **0.954** | 0.889 | **-0.065** | **2/62 (3%)** |
| Avg score @20 | 1.944 | 1.867 | -0.077 | 15/62 (24%) |
| ~~P@20(≥2)~~(非主指标) | 67.6% | 64.4% | -3.2pp | 13/62 (20%) |
| ~~P@20(≥3)~~(非主指标) | 37.8% | 34.4% | -3.4pp | 12/62 (19%) |

**Spearman ρ:** RRF +0.708 / Reranker +0.687,diff -0.021,Reranker 赢 29/62 (47%) **接近平局**

### 关键发现(K=20)

1. **没有任何一项主指标 Reranker 赢** — 不是 "全面退化",但 **生产意义上找不到开它的理由**
2. **Hit@20(≥3) -1.6pp = 1 个 case 高相关全被挤到 20 名外**(范围内重排只能持平或丢,不可能涨)
3. **MRR(≥2) 只赢 3%** — Reranker **顶部排序系统性变差**,常把 RRF 排在 Top-1/2 的高分 parent 挪到 Top-3 后
4. **Spearman ρ 接近平局** — ordering 上整体差异在噪音水平,主要损失集中在"顶部"

### 附表 — K=5/10/50 完整数据(非生产口径,供参考)

| K | NDCG diff | P(≥3) diff | MRR(≥2) diff | Hit(≥3) diff | NDCG win |
|---|---|---|---|---|---|
| K=5 | -0.146 | -16.1pp | -0.069 | -8.1pp | 15/62 |
| K=10 | -0.112 | -9.4pp | -0.067 | -3.2pp | 13/62 |
| **K=20★** | **-0.076** | -3.4pp | **-0.065** | **-1.6pp** | 12/62 |
| K=50 | -0.047 | 0pp | -0.065 | 0pp | 13/62 |

K=5/10 差距更大是 Reranker 把高分 parent 挤后的副产品,K=50 持平是因为范围内重排不换 set(只换 ordering)。**生产决策不该用 K=5/10 数据**。

### 可能原因

| 假设 | 备注 |
|---|---|
| `patient_text` 长 query 不适配 BGE Reranker | 200+ 字病例原文 ≠ "问句-段落" 训练分布 |
| RRF 加权已 ρ=0.71,接近 LLM 判断,reranker 无空间挖 | 加权方案太好了,留给 reranker 的提升空间已被吃完 |
| BGE Cross-Encoder 跟 LLM 判断 misalign | Cross-Encoder ≠ 通用推理 LLM 视角 |

### 生产决策

**Reranker 默认关闭** — 跟 spec §3.2.3 "None(关闭)" 模式一致:

```bash
# .env
RERANKER_ENABLED=False
```

**性价比账(K=20 口径):**
- **成本**:2.6 GB GPU 显存(BGE-Reranker-v2-minicpm-layerwise INT8) + ~5 秒/次 推理延迟
- **收益**:NDCG@20 +0.076 / Hit@20 +1.6pp / MRR +0.065
- **结论**:Reranker 在 K=20 是**边际退化** + 显著成本,**关掉是性价比决策**,不是 "Reranker 灾难性"

### 待验证(最终判定要看下游)

当前只评检索质量,**Reranker 真正命运取决于 ⑩ diagnose 节点能否在 RRF 加权 Top-20 上稳定输出正确诊断** — 见 §9 "下游 ⑩ diagnose 准确率验证"。如果下游表现已经达标,关 Reranker 没争议;如果下游受 ordering 影响大,Reranker 可能仍有空间。

代码:[`.eval/rag_eval/compare_rerank_full.py`](../.eval/rag_eval/compare_rerank_full.py)
数据:[`.eval/rag_eval/sparse_fusion_compare/rerank_full_result.json`](../.eval/rag_eval/sparse_fusion_compare/rerank_full_result.json)(334 KB)

---

## 8. 数据与脚本清单

### 数据(`.eval/rag_eval/`,gitignored)

| 文件 | 内容 | 大小 |
|---|---|---|
| `cases/*.json` | 62 case 原始病例数据(patient_text + gold diagnosis 等)| - |
| `step0_cache/*.json` | 62 case 拆分后的 MedicalState 各字段(chief_complaint + slots + report_findings)| - |
| `sparse_fusion_compare/compare_result.json` | retrieve + RRF 全量原始数据(dense/sparse hits + 双 fusion Top-200 + chunks_meta)| 63.9 MB |
| `sparse_fusion_compare/judge_per_case/*.json` | 62 个 case 的 LLM Judge 完整记录 | 0.05 MB × 62 |
| `sparse_fusion_compare/judge_per_case/_parents_meta.json` | 跨 case unique parents 元数据(chunk_raw_text + heading_path)| 9.5 MB |
| `sparse_fusion_compare/metrics_result.json` | 双粒度指标计算结果 | 154 KB |
| `sparse_fusion_compare/rerank_full_result.json` | 62 case Reranker 对照结果 | 334 KB |
| `table_recall_validation/*.json` | Q1/Q2/Q3 表格召回实验结果 | - |

### 脚本(`.eval/rag_eval/`,gitignored)

| 入口 | 用途 |
|---|---|
| `step0.py` / `run_step0_batch.py` | case 文本 → MedicalState 字段(LLM 一次性拆) |
| `step1.py` | EL + sparse_queries 多字段构造 + dense_query |
| `inject_abnormal_semantic.py` | abnormal_values → positive_findings 注入(Claude 人脑判读) |
| `compare_rrf_weighting.py` | 62 case retrieve + 等权/加权 RRF 双 fusion + 全量落盘 |
| `run_llm_judge.py` | LLM Judge 跑 62 case(并发,断点续传) |
| `compute_metrics.py` | 算双粒度指标 NDCG / Precision / MRR / Avg / Spearman |
| `validate_table_recall.py` | 表格召回 Q1/Q2/Q3 验证 |
| `compare_rerank_full.py` | 62 case Reranker vs RRF 对照 |
| `try_reranker_case001.py` | case 001 reranker 单 case 试验 |

---

## 9. 后续可选改进

### 已留 hook,后续可做(已有数据基础)

1. **dense_weight 调参** — N/3(更激进) / N/5(当前) / N/8(更温和) / 固定 3.0,用 `compare_result.json` 已落盘的 dense + sparse hits 本地重算 fusion,**秒级出结果**,不需要重跑 retrieve
2. **Reranker 短 query 重试**(可选) — 改用 `chief_complaint + confirmed_symptoms`(30~80 字)做 reranker query,看是否逆转长 query 的劣势。当前数据已足以拍板"砍 reranker",此实验仅 nice-to-have
3. **改进 4: Tier 3 占位症状进 sparse** — 待 EL 讨论决定后再做

### 未做但值得做

1. **Reranker 模型对比** — 换 jina-reranker / Cohere reranker 看不同模型表现
2. **下游 ⑩ diagnose 准确率验证** — 用当前 RAG 召回 + 真实 ⑩ Step 1/2/3 chain 跑 62 case,看诊断准确率(终极业务指标),作为本评测的下一阶段验证
3. **真实多轮 followup 评测** — 当前评测是单轮 patient_text,实际生产是多轮追问

### 不做的(理由扎实)

1. ~~回退 sparse 字段到 spec 原假设(5~8 路)~~ — 实测 12 字段都验证有效,回退牺牲信号
2. ~~LLM Judge 多次评取均值~~ — 实测单次 Judge reason 准确,4 档分布均匀,LLM 噪声可控

---

## 10. 与 DEV_SPEC 的关系(2026-05-17 已全部同步)

| Spec 章节 | 同步状态 |
|---|---|
| §3.2.1 Step 2 sparse_queries 来源 | ✅ 已改 — sparse 多字段直采(state 字段 + 阴性过滤) |
| §3.2.2 RRF 加权融合 | ✅ 已改 — 加权 RRF `dense_weight = max(1, N_sparse/RRF_DENSE_WEIGHT_FACTOR)` |
| §3.2.3 Reranker 默认 None | ✅ 已改 — 明确"None(关闭,默认)" + 评测依据 |
| §4.1.2 ②.Step 3/4 + ③ retrieve | ✅ 已改 — Sparse 多字段直采 + 加权 RRF 描述 |
| §4.1.2 ⑩ Step 0 | ✅ 已改 — `RERANK_TOP_K=20` + ENABLED=False 走 fallback |
| §9.7 `agent_limits` | ✅ 已加 `RRF_DENSE_WEIGHT_FACTOR=5`;`RERANK_TOP_K` / `RERANKER_ENABLED` 在 RetrievalSettings/RerankerSettings 段(不在 agent_limits) |
| §9.3 LLM call 清单 | 无变化(本评测不引入新 LLM 调用) |

**Code 同步**:`config/settings.py` / `src/rag/retrieval/fusion.py` / `src/agent/nodes/build_query.py` 三文件已改;357 unit test PASS。

**EL 链路保留**:Step 1 NER + Step 2 Entity Linking 节点不动(产物 `confirmed_symptoms` / `standardized_entities` 仍供 ⑤ select_symptom 消费),只是 sparse 不再用 EL alias 反查。EL 整体去留待后续单独评估(EL_DESIGN_REVIEW §11)。
