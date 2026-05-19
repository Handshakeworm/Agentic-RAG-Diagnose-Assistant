"""tests/unit/test_sparse_retriever.py — DEV_SPEC §3.2.2 Sparse Route 单测。

monkeypatch 替换底层 BM25 调用,验证:
- N 个维度产生 N 次 BM25 调用(顺序与 sparse_queries 一致)
- top_k 默认取 settings.agent_limits.RETRIEVE_TOP_N
- top_k 显式传值时透传到底层
- source_id_filter 透传
- 空入入 → 空列表(无副作用)
- 各路结果保留为 list[list[dict]] 形态,顺序与输入维度对齐
"""
from __future__ import annotations

import pytest

import src.rag.retrieval.sparse_retriever as sr
from config.settings import settings


@pytest.fixture
def fake_bm25(monkeypatch):
    """记录每次调用的 (query_text, top_k, source_id_filter) 并返回固定形态结果。"""
    calls: list[dict] = []

    def _fake_search(query_text: str, top_k: int = 20, source_id_filter: str | None = None):
        calls.append({
            "query_text": query_text,
            "top_k": top_k,
            "source_id_filter": source_id_filter,
        })
        # 返回伪数据:每路 1 条命中,包含 query_text 便于断言
        return [{
            "id": f"hit_for_{query_text}",
            "source_chunk_id": f"chunk_{query_text}",
            "vector_type": "original",
            "original_content": f"内容包含 {query_text}",
            "source_id": "fake_source",
            "score": 1.0,
        }]

    monkeypatch.setattr(sr, "search_sparse_bm25", _fake_search)
    return calls


def test_n_dims_produce_n_bm25_calls_in_order(fake_bm25) -> None:
    """3 个维度 → 3 次 BM25 调用,顺序与 sparse_queries 一致。"""
    queries = ["腹痛 肚子疼", "发热 发烧", "呕吐 想吐"]
    results = sr.search_sparse_routes(queries)

    assert len(fake_bm25) == 3
    assert [c["query_text"] for c in fake_bm25] == queries
    # 输出顺序与输入维度对齐
    assert len(results) == 3
    assert results[0][0]["original_content"] == "内容包含 腹痛 肚子疼"
    assert results[2][0]["original_content"] == "内容包含 呕吐 想吐"


def test_default_top_k_is_retrieve_top_n(fake_bm25) -> None:
    """top_k 不传时取 settings.agent_limits.RETRIEVE_TOP_N(权威 §9.7)。"""
    sr.search_sparse_routes(["腹痛"])
    assert fake_bm25[0]["top_k"] == settings.agent_limits.RETRIEVE_TOP_N


def test_explicit_top_k_overrides_default(fake_bm25) -> None:
    """显式传 top_k=50 → 底层每次都收到 50。"""
    sr.search_sparse_routes(["腹痛", "发热"], top_k=50)
    assert all(c["top_k"] == 50 for c in fake_bm25)


def test_source_id_filter_propagates_to_each_call(fake_bm25) -> None:
    """source_id_filter 透传给每次 BM25 调用(支持 E6 元数据 pre-filter)。"""
    sr.search_sparse_routes(["腹痛", "发热"], source_id_filter="src_abc")
    assert all(c["source_id_filter"] == "src_abc" for c in fake_bm25)


def test_empty_queries_returns_empty_list(fake_bm25) -> None:
    """空入入 → 空列表,无 BM25 调用(沿袭 build_query Step 2 长度过滤已剔空词袋的契约)。"""
    out = sr.search_sparse_routes([])
    assert out == []
    assert fake_bm25 == []


def test_returns_list_of_lists_shape(fake_bm25) -> None:
    """形态:外层 list[len = N], 内层 list[dict]。"""
    queries = ["腹痛", "发热"]
    out = sr.search_sparse_routes(queries)
    assert isinstance(out, list)
    assert all(isinstance(route, list) for route in out)
    assert all(isinstance(hit, dict) for route in out for hit in route)
