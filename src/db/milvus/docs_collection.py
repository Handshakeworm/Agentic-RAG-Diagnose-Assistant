"""docs_collection — Milvus 医学文献 chunk 多向量库操作(DEV_SPEC §2.4.1)。

提供建库 / 写入 / 双路检索(Dense + Sparse BM25)的最小操作集,
供 src/rag/ingestion/ 灌库与 Agent ③ retrieve 召回调用。
schema 与索引参数集中在 config/milvus_schema.py。

幂等约定:写入路径用 upsert(基于确定性主键 `id`,见 §2.4.1 id 规则表),
重跑 ETL 不会产生重复记录。
"""

from __future__ import annotations

import os
import time

from pymilvus import AnnSearchRequest, Collection, connections, utility

from config.milvus_schema import (
    DOCS_COLLECTION_NAME,
    DOCS_DENSE_INDEX,
    DOCS_SCALAR_INDEXES,
    DOCS_SCHEMA,
    DOCS_SPARSE_INDEX,
)
from src.common.metrics import milvus_rpc_latency_seconds


def _ensure_connection(alias: str = "default") -> None:
    if connections.has_connection(alias):
        return
    connections.connect(
        alias=alias,
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530")),
    )


def ensure_docs_collection(drop_existing: bool = False) -> Collection:
    """建表 + 建全部索引(dense HNSW + sparse BM25 + 3 个 scalar INVERTED),幂等。

    drop_existing=True 时先删后建(重灌库用)。
    """
    _ensure_connection()

    if utility.has_collection(DOCS_COLLECTION_NAME):
        if not drop_existing:
            return Collection(DOCS_COLLECTION_NAME)
        utility.drop_collection(DOCS_COLLECTION_NAME)

    coll = Collection(name=DOCS_COLLECTION_NAME, schema=DOCS_SCHEMA)
    coll.create_index(**DOCS_DENSE_INDEX)
    coll.create_index(**DOCS_SPARSE_INDEX)
    for idx in DOCS_SCALAR_INDEXES:
        coll.create_index(**idx)
    return coll


def upsert_chunks(records: list[dict]) -> int:
    """幂等批量写入。

    records 每项需含 7 业务字段:`id`、`source_chunk_id`、`vector_type`、
    `dense_vector`、`text_for_bm25`、`original_content`、`source_id`。
    `bm25_sparse` 由 BM25 Function 自动派生,**不要手动传入**。

    summary / question 记录的 `text_for_bm25` 必须是空串(不参与 BM25,见 §2.4.1)。
    """
    coll = ensure_docs_collection()
    result = coll.upsert(records)
    coll.flush()
    return result.upsert_count


def search_dense(
    query_vector: list[float],
    top_k: int = 20,
    source_id_filter: str | None = None,
    vector_type_filter: str | None = None,
) -> list[dict]:
    """Dense 向量 ANN 检索(单次,§3.2.2)。

    返回每个 hit 的核心字段(无去重,留给上层 RRF 融合或多向量去重)。
    可选过滤:
    - source_id_filter:按来源文档过滤
    - vector_type_filter:如只搜 "original"(BM25 也只在 original 上)
    """
    coll = ensure_docs_collection()
    coll.load()

    expr_parts: list[str] = []
    if source_id_filter:
        expr_parts.append(f'source_id == "{source_id_filter}"')
    if vector_type_filter:
        expr_parts.append(f'vector_type == "{vector_type_filter}"')
    expr = " and ".join(expr_parts) if expr_parts else None

    # HNSW 硬性约束:ef ≥ k(limit),否则 Milvus 直接报 "ef should be larger than k"。
    # 64 作为下限保留小查询(top_k≤32)的原默认行为;大 top_k 时按 2× 扩展候选池
    # 兼顾召回质量(spec §3.2.2:RETRIEVE_TOP_N=200,这里 ef=400)。
    ef = max(64, top_k * 2)
    t0 = time.perf_counter()
    try:
        raw = coll.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": ef}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "source_chunk_id", "vector_type", "original_content", "source_id"],
        )[0]
    finally:
        milvus_rpc_latency_seconds.labels(collection="docs", operation="search_dense").observe(time.perf_counter() - t0)

    return [
        {
            "id": hit.entity.get("id"),
            "source_chunk_id": hit.entity.get("source_chunk_id"),
            "vector_type": hit.entity.get("vector_type"),
            "original_content": hit.entity.get("original_content"),
            "source_id": hit.entity.get("source_id"),
            "score": hit.score,
        }
        for hit in raw
    ]


def search_sparse_bm25(
    query_text: str,
    top_k: int = 20,
    source_id_filter: str | None = None,
) -> list[dict]:
    """BM25 全文检索(§3.2.2 Sparse Route)。

    Milvus 2.4+ 内置:把 query_text 喂给 bm25_sparse 字段,
    会自动经 analyzer + BM25 函数转换为查询稀疏向量后检索。

    BM25 仅检索 original 记录(summary/question 的 text_for_bm25 是空串,
    自然不在倒排索引里),所以无需显式 vector_type 过滤。
    """
    coll = ensure_docs_collection()
    coll.load()

    expr = f'source_id == "{source_id_filter}"' if source_id_filter else None

    t0 = time.perf_counter()
    try:
        raw = coll.search(
            data=[query_text],
            anns_field="bm25_sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "source_chunk_id", "vector_type", "original_content", "source_id"],
        )[0]
    finally:
        milvus_rpc_latency_seconds.labels(collection="docs", operation="search_sparse_bm25").observe(time.perf_counter() - t0)

    return [
        {
            "id": hit.entity.get("id"),
            "source_chunk_id": hit.entity.get("source_chunk_id"),
            "vector_type": hit.entity.get("vector_type"),
            "original_content": hit.entity.get("original_content"),
            "source_id": hit.entity.get("source_id"),
            "score": hit.score,
        }
        for hit in raw
    ]


def count_chunks() -> int:
    coll = ensure_docs_collection()
    coll.flush()
    return coll.num_entities


def drop_docs_collection() -> None:
    _ensure_connection()
    if utility.has_collection(DOCS_COLLECTION_NAME):
        utility.drop_collection(DOCS_COLLECTION_NAME)
