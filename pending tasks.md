# pending tasks
# 第4章 Agent 设计 — 审阅问题清单


缺陷 3（安全风险，**⏸️ 暂搁置待后续完善**，DEV_SPEC.md §4.1.2 Node ⑪ 已加 TODO 注释）：safety_gate ⑪ 声称"确定性规则过滤"但实际使用 RAG
规范写道：规则层通过 RAG 检索对应 chunk 进行匹配判断，同时又将其定性为"规则过滤（确定性）"。

RAG 是概率性检索，若相关 chunk 未被召回，过敏药物/配伍禁忌过滤规则就不会触发。这对于安全关键的用药约束而言是根本性的架构矛盾——声称确定性但实现是概率性的，可能导致已知禁忌药物漏检。






# 基础设施

###  1. RAG 响应缓存 key 设计与 Agentic 场景不兼容（中影响）

5.1 节缓存设计：`rag:<hash(query_text)>` 缓存完整 RAG 响应，TTL 1h。

在 Agentic 工作流中，同一个 query 字符串在不同患者、不同追问轮次（不同的 `confirmed_symptoms`/`denied_symptoms`/`report_findings`）下，应召回并推理出不同的结果。用纯 query 字符串做 key，会导致不同患者/不同阶段命中同一缓存，返回错误的历史诊断结论。

建议：Agentic 工作流不适合做响应级缓存；若要缓存，最多缓存向量检索结果（`retrieve` 层），且 key 需包含完整上下文摘要哈希。
