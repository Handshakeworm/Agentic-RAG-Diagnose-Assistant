# Pending Tasks(跨章节待办 backlog)

> DEV_SPEC §8.4 进度表之外的「已识别但暂缓」事项。CLAUDE.md 引用本文件。
> 格式:背景 → 决策 → 实施约束。完成后移除并回写对应 spec 小节。

---

## 1. ⑩ Step 0/0.5:截断改到「父块层」+ 父块去重(喂 20 个 distinct 父块)

**背景**:当前 Step 0 在**子块层**截 `RERANK_TOP_K=20` → Step 0.5 才扩父块,且只对图表去重(规则 2/3),**规则 1 的父块文本不去重**。于是 top-20 子块里撞父时,同一段父块全文被重复拼进 `[文本块 N]`(`diagnose.py::_build_diagnose_context` line 263 + `prompts/agent.py::build_diagnose_prompt` line 1054,两层都没对父块去重)。spec §3.2.3 line 1900「四条规则展开后按 chunk_id 去重」**本意覆盖父块,代码未实现** = bug。撞父概率不低(全库平均 ~2.5 子/父,top-K 头部更易聚同节)。

**决策**(2026-06-06):截断**挪到父块层**——按 RRF/精排顺序遍历子块,边扩父块边按 `parent_chunk_id` 去重,凑满 **20 个 distinct 父块**即停。比"把 RERANK_TOP_K 拍大再去重"稳:后者 distinct 父块数随撞父率漂(18~22),父块层截断永远精确 20。理论上多喂 distinct 父块只会提升诊断性能。

**实施约束**:
1. 改 `diagnose.py` Step 0/0.5 + `config/settings.py`:新增 `DIAGNOSE_PARENT_TOP_K=20`(distinct 父块,§9.7);子块扫描设上限(~60 封顶,控 reranker 开时成本;reranker 默认关 = RRF 原序遍历近零成本)。
2. **必须重跑 `diagnose eval` 确认 top1 93.5% 不回退**——这是改生产诊断输入量(~12-15 → 20 distinct 父块),旧 eval 未覆盖此配置(以最新实验为准纪律)。
3. 同步回写 spec §3.2.3(收口"父块也去重"措辞)+ §4.1.2 ⑩ Step 0.5 描述。

---

## 2. ④ chunk-informed 选题(已记 DEV_SPEC §4.1.2 ④【Deferred】)

详见 spec 该节。摘要:④ 读 RRF 序 **top-5~8 distinct 父块全文**(复用任务 #1 的父块去重扩展,读原文不读 summary——鉴别特征是颗粒度的),选题转鉴别驱动,无新节点/无新 LLM 调用站点。**前置门槛**:先建「鉴别模糊」评测子集证明填槽式确有漏题再动;具体 K(5/8)实测定。依赖任务 #1 的父块去重落地。

---

## 3. 节点编号不一致(文档一致性)

④/⑤/⑥ 三处打架:CLAUDE.md(⑤ select_discriminative / ⑥ generate_followup)vs 代码(`select_symptom.py` docstring = ④)vs README 流程图(generate_followup 标 ⑤)。需以 `graph.py` 实际 `add_node` 拓扑为权威定一套编号,统一 CLAUDE.md + 两版 README + spec。留作「集中同步」批量做。

---

## 4. CLAUDE.md 悬空引用(本文件已建 → 已消解)

CLAUDE.md 引用 `pending tasks.md`,此前文件不存在。本文件重建后引用恢复有效。无需再动,留痕备查。

---

## 5. 灌注图 chunks 表行数(文档一致性)

README 灌注图节点「chunks 表 26054 行」不准:26054 是**已向量化子块数**,chunks 表实际 **34830 行**(26054 done + 8776 parent skip)。两改法:**A** = 写 `34830 行`;**B** = 写 `26054 embedded + 8776 parent`(摊开,跟 embedding 边 `child+table+figure` / parent skip 一一对应)。中英文 README 各一处。倾向 B。
