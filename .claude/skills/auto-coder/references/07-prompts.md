# 7. Prompt 模板

所有 LLM 调用的 Prompt 集中维护于 `src/prompts/`，按调用场景拆分为四个模块。

## 7.1 模块总览

| 文件 | 覆盖场景 | Prompt 数量 |
|------|---------|------------|
| `ingestion.py` | 数据摄取增强 | 1 |
| `agent.py` | Agent 节点（节点①–⑬ + compact_context 预留） | 17 |
| `evaluation.py` | LLM Judge | 7 |

> 注：查询预处理（3.2.1）的所有步骤均在 Agent ② `build_query` 节点内完成，对应 prompt 是 `src/prompts/agent.py` 的 `build_query_construction_prompt`，不再单独维护 `src/prompts/retrieval.py`。

## 7.2 各模块详细说明

### `ingestion.py` — 数据摄取增强

| 函数 | 说明 |
|------|------|
| `build_chunk_enrichment_prompt` | 对单个 chunk 一次性生成 title / summary / hypothetical_questions（合并为单次 LLM 调用，降低成本） |

> 注：知识库摄取管道不对图像类内容做 Vision LLM 理解（见 3.1.1 末"关于图像内容理解的设计原则"），因此不设 `build_image_caption_prompt`。患者检查报告的多模态解析 prompt 由 `build_exam_report_reading_prompt`（agent.py）承担。

### `agent.py` — Agent 节点

| 函数 | 对应节点 | 说明 |
|------|---------|------|
| `build_info_collect_prompt` | ① | 从 patient_input 提取主诉 + 现病史自由文本 + 现病史结构化槽位（`present_illness_slots`，13 个维度同步填充） |
| `build_exam_report_reading_prompt` | ①⑨ | 多模态理解检验单/影像报告（文字+图像+PDF），返回结构化摘要 |
| `build_ner_prompt` | ② | 从新增文本中抽取医疗实体（症状/疾病/药物/解剖），含否定标记与时序 |
| `build_query_construction_prompt` | ② | 基于标准化实体构造 Dense / Sparse 双路查询 |
| `build_smart_followup_prompt` | ⑤ | 1 LLM 直接选追问 — 输入 state(主诉 + 13 维 slots 空缺 + 已问症状),输出 `questions: list[FollowupQuestion]`(slot 维度填补 / open 兜底问) |
| `build_followup_question_prompt` | ⑥a | 将两种 type 追问项(slot 维度填补 + open 开放式)转化为患者可理解的流畅追问句式 |
| `build_followup_parse_prompt` | ⑦ | 解析患者追问回答:slot 类 → 回填 `present_illness_slots` + 追加 `present_illness`;open 类 → 提取新症状到 `new_symptoms`(由 ⑦ append 到 confirmed_symptoms) |
| `build_exam_recommendation_prompt` | ⑧ | 根据待鉴别症状推断所需检查（体格检查+辅助检查），输出优先级与鉴别理由 |
| `build_diagnose_prompt` | ⑩ 1 步 LLM | 诊断推理：全量患者画像(主诉+现病史+slots+symptoms+history+reports)+ 文献(20 父块 + figure 多模态)+ ⑤ unaskable 粗筛 → 一次 LLM 出 `DiagnosisOutput`(results + retained_unaskable);对齐 RAG 评测 `.eval/rag_eval/run_diagnose_eval.py` 口径,3 步链已废弃 |
| `build_safety_gate_prompt` | ⑪ | 规则层无法覆盖时的 LLM 兜底：交叉过敏风险、罕见药物相互作用、肝肾功能剂量调整 |
| `build_advice_prompt` | ⑫ | 在安全约束范围内生成用药建议 / 检查建议 / 风险提示，高风险路径（疑似心梗/卒中）优先输出急诊提示 |
| `build_format_response_prompt` | ⑬ | 将结构化诊断与建议整理为自然语言，附加免责声明 |
| `build_context_compression_prompt` | compact_context | 紧急 Compaction 第二步：将旧消息区压缩为结构化摘要，保留对话语境和推理过程（4.2.4 节，token 达 75% 阈值时触发） |

### `evaluation.py` — LLM Judge

| 函数 | 评估层 | 说明 |
|------|-------|------|
| `build_rag_faithfulness_prompt` | RAG 层 | 检索忠实度：最终回答中每个陈述是否能在召回 chunk 中找到依据（对齐 RAGAS Faithfulness） |
| `build_rag_relevance_prompt` | RAG 层 | 检索相关性：最终回答与原始查询的切题程度（对齐 RAGAS Answer Relevancy） |
| `build_hallucination_check_prompt` | Agent 层 | 幻觉检测：逐条核查 Agent 结论是否有上下文证据支撑，输出无依据结论占比（6.2.2 节） |
| `build_decision_trace_prompt` | Agent 层 | 追问决策合理性：从区分度、必要性、优先级三个子维度评分（各 1-5 分），并对证据链完整性打分（6.2.2 节） |
| `build_response_quality_prompt` | E2E 层 | 端到端响应质量综合评分（准确性、完整性、安全性）（6.3 节） |
| `build_advice_completeness_prompt` | E2E 层 | 建议完整性：用药建议 / 检查建议 / 高风险警告是否齐全（6.3 节） |
| `build_patient_simulation_prompt` | E2E 层 | LLM 模拟患者：以 patient_profile 为系统 prompt，扮演患者实时回答 Agent 追问，用于自动化多轮 E2E 测试（6.3.1 节） |

## 7.3 设计原则

- **封装形式**：Prompt 以 Python 函数封装，接受结构化参数，返回 `(system: str, user: str)` 元组，不在业务代码中内联字符串
- **导入方式**：`from src.prompts.agent import build_diagnose_prompt`，各模块职责边界清晰
- **可测试性**：Prompt 函数可独立单元测试，验证模板渲染正确性与参数边界行为
- **Few-shot 管理**：安全门控、诊断推理等高风险 Prompt 的 few-shot examples 与函数定义放在同一文件中，不得散落在业务代码里
- **版本追踪**：每个文件通过模块级 `__prompt_version__` 常量标记版本号，确保评估报告可回溯到具体 Prompt 版本




