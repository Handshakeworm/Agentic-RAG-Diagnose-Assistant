"""tests/unit/test_diagnose_context.py — ⑩ diagnose Context 扩展 3 规则(DEV_SPEC §3.2.3)。

覆盖:
- 规则 1:child → parent_chunk_id 父块全文替换
- 规则 1 兜底:parent_chunk_id 缺失 → 小块原文兜底
- 规则 2:table/figure 直接命中 → 进 figures + image_data_uri 加载
- 规则 3:父块 heading_path_id → 同节图表回拉(封顶 RETRIEVE_PARENT_FIGURE_CAP)
- 跨规则 2/3 去重:同 chunk_id 一份
- chunks_lookup 异常路径降级

(原规则 4 vector_hints 已删除 — ⑩ 重设计为 1 步 LLM 后,父块全文够用,vector_hints 冗余)
"""
from __future__ import annotations

from unittest.mock import patch


# ────────────────────────────────────────────────────────────────────────────
# 规则 1:child → 父块
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.lookup_figures_by_heading_path", return_value={})
@patch("src.agent.nodes.diagnose.lookup_chunk_content")
def test_rule1_child_expands_to_parent(mock_lookup, _figs):
    from src.agent.nodes.diagnose import _build_diagnose_context

    mock_lookup.side_effect = [
        # 第 1 次:查 child
        {
            "c_child": {
                "chunk_raw_text": "小块原文",
                "medical_statement": None,
                "parent_chunk_id": "c_parent",
                "heading_path_id": "hp1",
                "chunk_type": "child",
                "image_path": None,
                "summary": None,
                "title": None,
            }
        },
        # 第 2 次:查父块
        {
            "c_parent": {
                "chunk_raw_text": "父块整段全文 - 含完整临床上下文",
                "medical_statement": None,
                "parent_chunk_id": None,
                "heading_path_id": "hp1",
                "chunk_type": "parent",
                "image_path": None,
                "summary": None,
                "title": None,
            }
        },
    ]

    chunks = [{"source_chunk_id": "c_child", "vector_hits": []}]
    ctx = _build_diagnose_context(chunks, ["fallback"])

    assert ctx["parent_texts"] == ["父块整段全文 - 含完整临床上下文"]
    assert ctx["figures"] == []


@patch("src.agent.nodes.diagnose.lookup_figures_by_heading_path", return_value={})
@patch("src.agent.nodes.diagnose.lookup_chunk_content")
def test_rule1_no_parent_falls_back_to_self(mock_lookup, _figs):
    from src.agent.nodes.diagnose import _build_diagnose_context

    mock_lookup.return_value = {
        "c1": {
            "chunk_raw_text": "child 原文",
            "medical_statement": None,
            "parent_chunk_id": None,  # 无父块
            "heading_path_id": "hp1",
            "chunk_type": "child",
            "image_path": None,
            "summary": None,
            "title": None,
        }
    }

    ctx = _build_diagnose_context([{"source_chunk_id": "c1"}], ["fb"])
    assert ctx["parent_texts"] == ["child 原文"]


# ────────────────────────────────────────────────────────────────────────────
# 规则 2:figure/table 直接命中 → 进 figures + 加载 image_data_uri
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.load_report")
@patch("src.agent.nodes.diagnose.lookup_figures_by_heading_path", return_value={})
@patch("src.agent.nodes.diagnose.lookup_chunk_content")
def test_rule2_figure_direct_hit_loads_image(mock_lookup, _figs, mock_load):
    from src.agent.nodes.diagnose import _build_diagnose_context

    mock_lookup.return_value = {
        "fig1": {
            "chunk_raw_text": "图 2-4 急性胆囊炎超声",
            "medical_statement": "should NOT enter prompt",
            "parent_chunk_id": None,
            "heading_path_id": "hp1",
            "chunk_type": "figure",
            "image_path": "/data/medical-resources/figures/x.jpg",
            "summary": None,
            "title": "急性胆囊炎超声图",
        }
    }
    mock_load.return_value = {
        "kind": "image",
        "media_type": "image/jpeg",
        "data": "BASE64",
        "data_uri": "data:image/jpeg;base64,BASE64",
    }

    ctx = _build_diagnose_context([{"source_chunk_id": "fig1"}], ["fb"])
    assert len(ctx["figures"]) == 1
    f = ctx["figures"][0]
    assert f["chunk_id"] == "fig1"
    assert f["chunk_type"] == "figure"
    assert f["image_data_uri"] == "data:image/jpeg;base64,BASE64"
    # spec §3.1.5.1:medical_statement 不进 figures payload
    assert "medical_statement" not in f


@patch("src.agent.nodes.diagnose.load_report", side_effect=FileNotFoundError("missing"))
@patch("src.agent.nodes.diagnose.lookup_figures_by_heading_path", return_value={})
@patch("src.agent.nodes.diagnose.lookup_chunk_content")
def test_rule2_image_load_failure_yields_none_uri(mock_lookup, _figs, _load):
    from src.agent.nodes.diagnose import _build_diagnose_context

    mock_lookup.return_value = {
        "fig1": {
            "chunk_raw_text": "caption",
            "medical_statement": None,
            "parent_chunk_id": None,
            "heading_path_id": "hp1",
            "chunk_type": "figure",
            "image_path": "/missing.jpg",
            "summary": None,
            "title": None,
        }
    }

    ctx = _build_diagnose_context([{"source_chunk_id": "fig1"}], ["fb"])
    assert ctx["figures"][0]["image_data_uri"] is None  # 加载失败 → None,不抛


# ────────────────────────────────────────────────────────────────────────────
# 规则 3:父块 → heading_path_id 同节图表
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.load_report")
@patch("src.agent.nodes.diagnose.lookup_figures_by_heading_path")
@patch("src.agent.nodes.diagnose.lookup_chunk_content")
def test_rule3_parent_pulls_same_section_figures(
    mock_lookup, mock_figs, mock_load,
):
    from src.agent.nodes.diagnose import _build_diagnose_context

    mock_lookup.side_effect = [
        # 第 1 次:查 child
        {
            "c1": {
                "chunk_raw_text": "small",
                "medical_statement": None,
                "parent_chunk_id": "p1",
                "heading_path_id": "hp_chole",
                "chunk_type": "child",
                "image_path": None,
                "summary": None,
                "title": None,
            }
        },
        # 第 2 次:查父块
        {
            "p1": {
                "chunk_raw_text": "急性胆囊炎临床表现章节全文",
                "medical_statement": None,
                "parent_chunk_id": None,
                "heading_path_id": "hp_chole",
                "chunk_type": "parent",
                "image_path": None,
                "summary": None,
                "title": None,
            }
        },
    ]
    mock_figs.return_value = {
        "hp_chole": [
            {
                "chunk_id": "f_us",
                "chunk_type": "figure",
                "chunk_raw_text": "图 2-4 超声",
                "image_path": "/x.jpg",
                "title": "胆囊超声",
                "relative_chunk_index": "figure:p10_b1",
            },
            {
                "chunk_id": "t_tg18",
                "chunk_type": "table",
                "chunk_raw_text": "<table>TG18 分级</table>",
                "image_path": "/y.jpg",
                "title": None,
                "relative_chunk_index": "table:p12_b3",
            },
        ]
    }
    mock_load.return_value = {"kind": "image", "data_uri": "data:image/jpeg;base64,X"}

    ctx = _build_diagnose_context([{"source_chunk_id": "c1"}], ["fb"])
    assert ctx["parent_texts"] == ["急性胆囊炎临床表现章节全文"]
    figure_ids = {f["chunk_id"] for f in ctx["figures"]}
    assert figure_ids == {"f_us", "t_tg18"}
    # 调用时应传 cap = settings.agent_limits.RETRIEVE_PARENT_FIGURE_CAP
    from config.settings import settings
    mock_figs.assert_called_once()
    assert mock_figs.call_args.kwargs["cap"] == settings.agent_limits.RETRIEVE_PARENT_FIGURE_CAP


# ────────────────────────────────────────────────────────────────────────────
# 跨规则 2/3 去重
# ────────────────────────────────────────────────────────────────────────────


@patch("src.agent.nodes.diagnose.load_report", return_value={"kind": "image", "data_uri": "X"})
@patch("src.agent.nodes.diagnose.lookup_figures_by_heading_path")
@patch("src.agent.nodes.diagnose.lookup_chunk_content")
def test_rule2_3_dedup_by_chunk_id(mock_lookup, mock_figs, _load):
    """figure chunk 直接命中(规则 2)+ 规则 3 拉同节同图 → 只留一份。"""
    from src.agent.nodes.diagnose import _build_diagnose_context

    mock_lookup.return_value = {
        "fig_shared": {
            "chunk_raw_text": "fig caption",
            "medical_statement": None,
            "parent_chunk_id": None,
            "heading_path_id": "hp1",
            "chunk_type": "figure",
            "image_path": "/x.jpg",
            "summary": None,
            "title": None,
        }
    }
    # 规则 3 通过 heading_path_id="hp1" 又拉到同一张 fig_shared
    mock_figs.return_value = {
        "hp1": [
            {
                "chunk_id": "fig_shared",
                "chunk_type": "figure",
                "chunk_raw_text": "fig caption",
                "image_path": "/x.jpg",
                "title": None,
                "relative_chunk_index": "figure:p1_b1",
            }
        ]
    }

    ctx = _build_diagnose_context([{"source_chunk_id": "fig_shared"}], ["fb"])
    assert len(ctx["figures"]) == 1
    assert ctx["figures"][0]["chunk_id"] == "fig_shared"


# ────────────────────────────────────────────────────────────────────────────
# 空入入 / lookup 异常降级
# ────────────────────────────────────────────────────────────────────────────


def test_empty_input_returns_empty_context():
    from src.agent.nodes.diagnose import _build_diagnose_context
    ctx = _build_diagnose_context([], [])
    assert ctx == {"parent_texts": [], "figures": []}


@patch("src.agent.nodes.diagnose.lookup_figures_by_heading_path", return_value={})
@patch("src.agent.nodes.diagnose.lookup_chunk_content", side_effect=RuntimeError("PG down"))
def test_lookup_exception_degrades_to_reranked_text(_lookup, _figs):
    from src.agent.nodes.diagnose import _build_diagnose_context

    chunks = [{"source_chunk_id": "c1"}]
    ctx = _build_diagnose_context(chunks, ["原文兜底"])
    assert ctx["parent_texts"] == ["原文兜底"]
    assert ctx["figures"] == []
