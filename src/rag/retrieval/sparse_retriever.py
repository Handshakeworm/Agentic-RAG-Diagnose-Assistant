"""src/rag/retrieval/sparse_retriever.py — Sparse Route 多维度 BM25 召回(DEV_SPEC §3.2.2 Sparse Route)。

Agent ③ retrieve 节点 Sparse 路入口:接收 ② build_query 产出的 `sparse_queries`
(每项一个症状维度的别名词袋),对每项分别走一次 Milvus BM25,返回 N 个 Top-K
候选列表(N = 维度数),交由 ④ RRF 融合。

底层 `docs_collection.search_sparse_bm25` 已封装单次 BM25 检索;本模块只做循环
分发 + Top-K 默认值 + 元数据过滤透传。

**spec gap 备案**(§8.3 E2 验收说"各自返回 Top-N",但未明示 N 的具体值):
默认 `top_k = settings.agent_limits.RETRIEVE_TOP_N`(=200),与 RRF 融合后截断
名额一致。多向量聚合(§3.2.2)可能让等效 chunk 数缩水,调用方需要时自行放大。
"""
from __future__ import annotations

from config.settings import settings
from src.db.milvus.docs_collection import search_sparse_bm25


def search_sparse_routes(
    sparse_queries: list[str],
    top_k: int | None = None,
    source_id_filter: str | None = None,
) -> list[list[dict]]:
    """对 `sparse_queries` 中每个症状维度词袋分别跑 BM25,返回 N 路候选。

    Args:
        sparse_queries: ② build_query Step 2 产出的 list[str],每项一个非空有效词袋
            (空词袋已由 build_query Step 2 长度 ≥ 2 + 保序去重过滤,本函数不做
            truthy 兜底,空入入直接返 [])
        top_k: 每路返回 Top-K 候选数。None 时取 `settings.agent_limits.RETRIEVE_TOP_N`
        source_id_filter: 可选 pre-filter,按来源文档过滤(对接 E6 元数据过滤策略)

    Returns:
        list[list[dict]]:外层长度 = len(sparse_queries),保留输入维度顺序;
        内层每项是 docs_collection.search_sparse_bm25 的 Top-K 结果(命中 chunk
        含 id / source_chunk_id / vector_type / original_content / source_id / score)。
        BM25 仅命中 original 记录(summary/question 的 text_for_bm25 是空串,自然
        不在倒排索引里,见 §2.4.1 设计)。
    """
    if top_k is None:
        top_k = settings.agent_limits.RETRIEVE_TOP_N
    return [
        search_sparse_bm25(query_text=q, top_k=top_k, source_id_filter=source_id_filter)
        for q in sparse_queries
    ]
