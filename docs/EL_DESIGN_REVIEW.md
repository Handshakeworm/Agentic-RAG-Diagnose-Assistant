# Entity Linking 设计评审 — 是否砍 EL?

**状态**:讨论中,未决定
**作者**:Claude(用户问询触发)
**触发**:Step 0.5 EL 验证发现 Tier 2 全 Miss,引出 EL 维护成本 vs 实际命中率的疑问
**适用**:DEV_SPEC §3.2 / §4.1.2 / §4.1.6 / §9 多处直接受影响
**决策权**:用户

---

## 1. 现状速览

EL = Entity Linking,把"病人/教材里的症状词"归一化到 ICD-10 标准 concept。
当前实现([src/agent/nodes/build_query.py:_link_one_entity](../src/agent/nodes/build_query.py))三层归一化:

```
Tier 1  query_term_by_alias_exact()       精确别名匹配,confidence=1.0
Tier 2  embed.encode() + search_aliases() 向量检索 cosine ≥ 0.92
Tier 3  保留原文 raw_text,concept_id=None,confidence=0
```

依赖:
- Milvus `terms_collection` ─ 当前 40,474 条 alias(见 `pending tasks.md` D2 任务)
- 本地 `Qwen3-Embedding-8B` GPU 模型(8GB+ 显存)

## 2. EL 在系统里的 5 个真实职责

| # | 职责 | 创建于 | 消费方 |
|---|---|---|---|
| ① | **Sparse BM25 词袋扩展** | ② Step 2 EL → Step 3 反查别名 | Milvus BM25 sparse 检索 |
| ② | **已问/已答症状去重** | ② / ⑤ / ⑦ 维护 confirmed/denied | ⑤ `_filter_already_asked`, ⑥ followup prompt |
| ③ | **信息增益计算单位** | ④ extracted_symptoms 的 preferred_term | ⑤ 二元熵公式(算 P(候选病\|症状)) |
| ④ | **报告证据自动消费** | ④ extracted + ①.5 report_findings | ⑤ `_consume_report_evidence`(positive→confirmed,negative→denied) |
| ⑤ | **API 输出 / 审计** | ② standardized_entities | `/diagnosis` API,`rag_trace` 审计 |

## 3. 实测命中率参考(case 1 跑 Step 0.5 EL)

case 1 `associated_symptoms = ["恶心", "呕吐", "烦躁不安", "意识障碍"]`:

| 症状 | 链上 | Tier |
|---|---|---|
| 恶心 | R11xx02 | Tier 1 精确 |
| 呕吐 | R11xx01 | Tier 1 精确 |
| 烦躁不安 | (未链) | Tier 3 占位 |
| 意识障碍 | (未链) | Tier 3 占位 |

50% Tier 3 占位率。Tier 2 一个都没命中。原因:
- ICD-10 中文版 alias 表只收"标准术语 + 拉丁缩写",几乎不收口语描述词
- Tier 2 阈值 0.92 在中文 embedding 空间偏严(中文同义词向量相似度天然低于英文)

## 4. 替代方案(逐职责)

### ① Sparse 词袋扩展

**现路径**:`EL → concept_id → query_aliases_by_concept_id() → 词袋拼接`

**替代**:让 ② Step 4 LLM 同时输出 sparse_queries
```python
class QueryConstructionOutput(BaseModel):
    dense_query: str
    sparse_queries: list[str]  # 每条 = "症状原词 同义词1 同义词2 英文"
```
LLM 训练数据本身就有"腹痛=肚子疼=胃痛=abdominal pain"这种知识,**免维护 alias 表**。

**风险**:LLM 同义词扩展不稳定(每次调用结果可能略有不同)。
**缓解**:temperature=0,加 retry,基本可重复。

### ② 已问/已答去重

**现路径**:`set(preferred_term).contains(候选 preferred_term)`(Tier 1/2);Tier 3 用 embedding 软比对

**替代**:LLM 一次语义比对,统一处理
```
已问过的症状:[心慌, 胸闷, 气短]
候选待问:[心悸, 呼吸困难, 头痛]
任务:剔除跟已问语义同义的(心悸=心慌就剔)
```

**收益**:比 set 比对严密(LLM 能识别"心悸=心慌"而 EL Tier 3 软比对易误判),且 Tier 3 那段 numpy 余弦相似度代码可以删。

### ③ 信息增益单位 ★ 这是 EL 唯一的硬刚理由

**现路径**:用 preferred_term 作为可比 key 算二元熵 `H(疾病|症状)`,有 `ASKABLE_GAIN_THRESHOLD=0.15` 早退收敛保证

**替代 A**:LLM 主观打分
```
候选问题 = ["问发热吗?", "问咳嗽吗?", "问胸痛吗?"]
任务:每个问题打 0-1 分,代表"问完后能区分多少候选病"
```
- ✗ 失去统计可解释性
- ✗ 不稳定(同 prompt 不同跑分数浮动)
- ✓ 不需要预归一化

**替代 B**:⑤ 整个改成 LLM 直接选"问哪个最有诊断价值",跳过信息增益机制
- 接近人类医生决策风格
- 失去 ASKABLE_GAIN_THRESHOLD 早退保证(可能轮数跑超)
- 评测不可解释

**判断**:这一职责砍 EL 代价**最大**,需要单独验证。

### ④ 报告证据自动消费

**现路径**:`positive_findings 字符串 == preferred_term` → 自动加入 confirmed_symptoms

**替代**:LLM 一次比对
```
已知阳性发现:["低热", "咽痛", "WBC升高"]
候选追问症状:[发热, 颈淋巴结肿大, 咳嗽]
任务:候选里哪些已经被阳性发现覆盖?(低热⊂发热) → 加 confirmed
```
**收益**:LLM 能理解"低热"⊂"发热"的语义包含关系,字符串等于是做不到的。

### ⑤ API 输出 / 审计

**现路径**(澄清):`standardized_entities` **不在对外 API 返回里**,只写入内部审计表 `rag_trace.intent_result` JSONB 字段。
对外 `DiagnoseResponse`([src/api/schemas/diagnosis_schema.py:48-74](../src/api/schemas/diagnosis_schema.py#L48-L74))只返回 session_id / status / pending_question / recommended_tests / final_response / diagnosis_result / medication_advice / risk_warnings。

**砍 EL 的实际影响**:
- ✓ 前端 API 0 影响(对外 schema 本来不包含)
- ⚠️ `rag_trace.intent_result` JSONB 字段内容变成只有 raw_text 列表,不破坏表结构
- ⚠️ Prometheus 的 EL 命中率指标(如果有)需相应调整

**用户已确认**:不需要返回 ICD code → ⑤ 可砍。

## 5. 变动范围矩阵

| 文件 | 砍 EL 后操作 | 评估 |
|---|---|---|
| [src/agent/nodes/build_query.py](../src/agent/nodes/build_query.py) | 砍 `_link_one_entity` / `_link_entities` 函数,Step 2 整段删 | ~80 行 |
| [src/agent/nodes/extract_symptoms.py](../src/agent/nodes/extract_symptoms.py) | 砍 `_normalize_keyword`,改成 LLM 抽取 + 同义词聚合 | ~50 行重写 |
| [src/agent/nodes/select_symptom.py](../src/agent/nodes/select_symptom.py) | `_filter_already_asked` / `_consume_report_evidence` 改 LLM | ~40 行重写 |
| [src/rag/retrieval/query_processing.py](../src/rag/retrieval/query_processing.py) | 整个文件可以删(只服务 sparse 词袋扩展) | -67 行 |
| [src/db/milvus/terms_collection.py](../src/db/milvus/terms_collection.py) | 整个文件可以删 | -180 行 |
| [config/milvus_schema.py](../config/milvus_schema.py) | 删 terms_collection schema 段 | -30 行 |
| [terms/](../terms/) 目录 | 整个删(D2 任务建表脚本) | -数百行 |
| [src/agent/state.py](../src/agent/state.py) | `standardized_entities` 字段保留还是删?(取决于 ⑤) | 边界 |
| [src/agent/schemas/entity_linking.py](../src/agent/schemas/entity_linking.py) | 整个文件可以删 | -1 个文件 |
| [src/agent/schemas/info_collect.py](../src/agent/schemas/info_collect.py) | 不动(PresentIllnessSlots 跟 EL 无关) | 0 |
| DEV_SPEC §9.5 | 删 `EntityLinkingMatch` schema 条目 | spec 改 |
| DEV_SPEC §9.7 | 删 `ENTITY_LINKING_TIER2_THRESHOLD` 常量 | spec 改 |
| .env / config/settings.py | 删 `AGENT_ENTITY_LINKING_TIER2_THRESHOLD` | 配置改 |
| GPU 资源 | embedding model (Qwen3-Embedding-8B INT8 ~8.5GB) **仍需要**(C2/D 灌库还在用),但运行时检索不再独占 | GPU 容量 |

**注**:embedding model 砍不掉 — chunks 库 ingestion 阶段(C2/C3)仍要用 dense vector。terms_collection 砍掉后,query 时不再需要常驻加载,GPU 实时压力下降。

## 6. 总账

**每轮 followup 多 ~1 次 LLM 调用**(用 LLM 替代 EL 做去重+证据消费)
- 单 case 5 轮 ≈ +$0.025-0.05
- 评测 60 case 总 +$1.5~3
- 生产 100 user × 10 case/天 = +$2.5~5/天 — 量级可承受

**砍掉**:
- ✓ terms_collection 持续维护成本(D2 灌库脚本 + ICD-10 alias 表升级)
- ✓ Tier 1/2/3 三层逻辑代码(分散在 build_query / extract_symptoms / select_symptom)
- ✓ Tier 3 软比对的 numpy 余弦相似度逻辑
- ✓ 评测时"EL 命中率指标"这个监控维度
- ✓ DEV_SPEC §9.5 EntityLinkingMatch schema 维护

## 7. 推荐路径(分两步走,可逆)

> ⚠️ **2026-05-16 更新**:Step A/B 路径基于"EL 在 ③ 信息增益是硬刚理由"的假设;§11 实测推翻了这个假设的前提 — ④ extract_symptoms 当前 TF-IDF 设计**根本喂不出可比 key**,信息增益机制本身无法成立。看 §11 后再判断这一节是否还有意义。

### Step A:砍 ①②④⑤,留 ③

只动:Sparse 词袋扩展 / 去重 / 报告消费 / API 输出
保留:信息增益机制(③)继续用 EL,但 terms_collection 不再实时常驻 — 改成 ④ extract_symptoms 出 raw 词,⑤ 算信息增益时按需 EL 一次

变更量约 200 行代码;DEV_SPEC §9.5 / §9.7 改 2 处;评测前后对照看追问质量是否下降。

### Step B:基于 Step A 的评测数据,决定 ③ 是否也砍

如果 LLM 主观打分给出的"问哪个症状"跟二元熵给的差不多,且评测分数(诊断准确率、轮数收敛、可解释性)可接受 → 整个 EL 砍掉

如果 LLM 主观打分给出的问题明显劣于二元熵 → ③ 这一职责保留 EL,terms_collection 缩到只服务这个用途(可能只需几百条核心症状词,不是 4 万条)

## 8. 风险

1. **Tier 1 命中的高频词("恶心""呕吐""发热")会失去 concept_id 链** — 影响审计/前端 ICD 映射
2. **LLM 替代方案延迟略增** — 每轮 +1 LLM 调用 ≈ 2-5 秒
3. **API 输出 contract 变更** — `/diagnosis` 返回少了 concept_id 字段,前端可能要改
4. **失去"EL 命中率"作为系统观测指标** — Prometheus 那段要相应调整
5. **DEV_SPEC §9 多处改动** — Authority hierarchy 要谨慎,§9 是全局契约

## 9. 评审决策检查清单

请用户拍板时核对:

- [x] 产品定位是否需要返回标准 ICD code? — **用户:不需要** → ⑤ 可砍
- [ ] 评测时"轮数收敛"指标多重要? — **用户:不好说,要数据** → 留到 Step B 评测后定 ③
- [ ] 接受每轮 followup 多 ~$0.005 LLM 成本?
- [x] ~~接受 API output contract 变更?~~ — **不存在该变更**(对外 API 本就不含 standardized_entities,只内部审计表用)
- [ ] DEV_SPEC §9.5 / §9.7 改动节奏(立即 / 评测后 / 不改)
- [ ] 是否走"先 Step A,再 Step B"的分步路径?

## 10. 不做的事

- ✗ 不立即修改任何代码
- ✗ 不立即改 DEV_SPEC
- ✗ 等用户在 §9 决策检查清单上拍板后,再写实施 PR

---

## 11. 实测发现 — ④ extract_symptoms 设计本身不工作 (2026-05-16)

**触发**:用户提出"先抽几个 chunk 验证 TF-IDF + EL 实际效果如何,如果本来就是错误的设计,就不用纠结要不要删 EL 了"。

**验证脚本**:[.eval/rag_eval/validate_node4_tfidf_el.py](../.eval/rag_eval/validate_node4_tfidf_el.py)
- 从内科学 source `37ed0fa69150a841` 确定性 hash 抽样 50 个 child chunk
- 跑 [src/agent/nodes/extract_symptoms.py](../src/agent/nodes/extract_symptoms.py) 同款逻辑:`TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` 单 chunk 取 Top-30,再走 EL 三层(Tier 1 alias / Tier 2 vector ≥0.92 / Tier 3 占位)
- 统计:关键词性质、EL Tier 分布、能形成的"症状 key → chunk 集合"倒排表(③ 信息增益要的可比 key 实际成型情况)

### 11.1 共池 TF-IDF Top-30 关键词(50 chunk 一起算)

```
['治疗', '细胞', '呼吸', '病毒', '病人', '静脉', '中毒', '药物', '疾病',
 '血管', '手术', '症状', '引起', '功能', '表现', '肥胖', '通气', '动脉',
 '激素', '感染', '包括', '代谢', '性心', '检查']
```

**抽出的根本不是症状词,是医学教材通用高频词**:"治疗 / 细胞 / 病人 / 药物 / 症状 / 包括 / 检查 / 引起 / 功能 / 表现" — 完全不是"右上腹放射痛""持续性钝痛"这类鉴别诊断信号。

原因:整本内科学每章都讲治疗、都提细胞,IDF 视角下这些词压不下去;反而"鉴别症状词"在单章节集中讨论,跨 chunk 稀疏,TF-IDF 分数反而低。

### 11.2 char_wb n-gram 大量碎片污染

实测样例:
- chunk 1(下丘脑疾病):抽出 `下丘 / 下丘脑 / 丘脑 / 时, / 受损 / 受损时 / 受损时,` — 同一词被 char n-gram 切碎 4 份 + 标点+常用词组合
- chunk 2(SAHS):抽出 `ah / sah / ahs / hs / sa / os / sahs` — SAHS 英文缩写被切成 7 段
- 还有 `病、` `等。` `病等` `性心` 这种标点 + 半个词的产物

`char_wb` 不带词典分词,在中文上必然产生这类副产品。

### 11.3 EL Tier 分布:93.9% 走 Tier 3

| Tier | 数量 | 占比 |
|---|---|---|
| Linked (Tier 1 alias + Tier 2 vector ≥0.92) | 78 | **6.1%** |
| Tier 3 占位(EL 无能为力) | 1192 | **93.9%** |

case 1 的 step 0.5 是 50% Tier 3;这里 KB chunks 上 94% Tier 3,**比已经规整化的 case 症状词糟两倍**。原因:KB 文本量大、领域术语密、n-gram 噪音多,40k 条 ICD-10 alias 表对教材实际文本覆盖远低于结构化症状词。

### 11.4 信息增益 (③) 要的可比 key 根本立不起来

```
Linked preferred_term 维度:     45 个 unique key
其中 ≥2 chunk 共享 (能做条件概率统计的):  9 个
```

50 chunk 抽完只得 **9 个 ≥2 chunk 共享的 preferred_term key**,且 Top-9 全是"癌 / 糖尿病 / 感染 / 肥胖 / 出血 / 疼痛"这种**诊断或状态词,不是鉴别症状**。

③ 信息增益要的是 `P(候选病 | 症状)` 这个条件概率分布,而要算这个分布需要"同一症状 key 在不同候选病的 chunk 里出现的频率分布"。当前产出:

- 抽出的关键词不是症状(95% 是通用领域词+碎片)
- 能形成可比 key 的只有 6.1%
- 这 6.1% 中绝大多数是诊断词(不能作为"待问症状")

**统计基础完全不成立** — 不是"EL 不够好",是 ④ 喂给 ⑤ 的数据本身就不是症状。

### 11.5 真问题改写

| 原命题 | 实测后命题 |
|---|---|
| 砍不砍 EL? | ④ 当前抽症状思路是否需要重设计? |
| EL 5 个职责中 ③ 信息增益是硬刚理由 | ③ 信息增益的"可比 key"需求,**当前根本没人给它喂** — EL 砍不砍跟 ③ 能否工作没关系 |
| Step A 砍 ①②④⑤ 留 ③ | 没意义 — 留下的 ③ 没有数据源喂 |

### 11.6 两条可行替代路径

**方案 A — ④ 改 LLM 抽症状(运行时)**

让 LLM 直读 reranked 后 Top-K=20 chunk 原文,一次性产出"该 chunk 提到的鉴别诊断症状清单"(已归一化标准词)。

- ✓ 抽真症状(LLM 知道"治疗"不是)
- ✓ 直接产出归一化的 preferred_term(LLM 知道"心慌=心悸",免 alias 表)
- ✗ +20 LLM 调用 / 轮,每轮 +$0.05-0.1
- ✗ 单 case 5 轮 ≈ +$0.25-0.5,评测 60 case ≈ +$15-30

**方案 B — chunk ingestion 时 LLM 一次性灌入 symptoms 字段**

在 C8 enrichment 阶段(现有 title / summary / tags / hypothetical_questions 旁边)让 LLM 多产出一个 `symptoms: list[str]` 字段。④ 节点变成"读 chunk 字段 + 合并",**零运行时 LLM 调用**。

- ✓ 长期最干净 + 零运行时成本
- ✓ §9.2 schema evolution 允许加新字段(带默认值兼容老数据)
- ✗ 现有库已灌完,需要回填 enrichment(C8 重跑 LLM,一次性成本)
- ✓ symptoms 字段一旦灌好,后续所有改动(EL 砍/留、③ 改造等)都不影响数据层

### 11.7 新决策树

```
④ 改 A (运行时 LLM)  → EL 可立即砍(LLM 直接出归一化症状)→ ⑤ 信息增益用 LLM 出的 preferred_term 做 key,可行
                    OR ⑤ 直接改 LLM 主观打分,跳过信息增益机制

④ 改 B (ingestion LLM) → EL 可立即砍(symptoms 字段已归一化)→ ③ ④ ⑤ 全链路无 EL 也工作
                       → 需要 C8 enrichment 加一个 LLM 调用 + 全量回填
```

**B 是终态最干净的方案**;A 是过渡期最快验证的方案。

### 11.8 待用户拍板

- [ ] 进 A 还是 B?(或两个都做:先 A 短期跑评测,B 列入下个 ingestion 周期)
- [ ] 进 B 的话,何时回填全量 chunks?(影响 ingestion GPU 排期)
- [ ] EL terms_collection 是否可以**立即**冻结(不再加 alias,不再调阈值),等替代路径落地后整体下线?
