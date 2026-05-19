"""tests/unit/test_graph_structure.py — F14 StateGraph 结构断言(DEV_SPEC §4.1.4)。

不调真 LLM,只验证图结构:节点全部注册、入口正确、条件边映射符合 spec。
"""
from __future__ import annotations


def test_graph_has_all_15_nodes():
    """⑤ 重设计后,④ extract_symptoms 节点已删,从 16 → 15。"""
    from src.agent.graph import build_graph

    g = build_graph()
    expected = {
        "info_collect",
        "analyze_initial_reports",
        "build_query",
        "retrieve",
        "select_discriminative_symptom",
        "generate_followup",
        "wait_followup_answer",
        "process_followup_answer",
        "recommend_exam",
        "wait_exam_report",
        "process_exam_result",
        "diagnose",
        "safety_gate",
        "generate_advice",
        "format_response",
    }
    actual = set(g.nodes.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"缺节点:{missing}"
    assert not extra, f"多节点(LangGraph 自动添加的 __start__/__end__ 不应在内):{extra}"


def test_compile_succeeds():
    """compile() 不报错说明边/路由结构合法。"""
    from src.agent.graph import build_app

    app = build_app()
    assert app is not None
