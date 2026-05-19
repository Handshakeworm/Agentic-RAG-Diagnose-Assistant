"""src/agent/graph.py — Agent LangGraph StateGraph 编排(DEV_SPEC §4.1.4)。

注册 15 节点 + 2 条件边,顺序与 4.1.3 流程图严格一致:

  ① info_collect → ①.5 analyze_initial_reports → ② build_query → ③ retrieve →
  ④ select_discriminative_symptom →
    ┌─[should_continue 路由]─┐
    │  followup → ⑤ generate_followup → ⑥ wait_followup_answer (interrupt) →
    │             ⑦ process_followup_answer → ② build_query (loop)
    └─ diagnose → ⑩ diagnose →
       ┌─[diagnose_router 路由]─┐
       │  recommend_exam → ⑧a recommend_exam → ⑧b wait_exam_report (interrupt) →
       │                   ⑨ process_exam_result → ② build_query (loop)
       └─ safety_gate → ⑪ safety_gate → ⑫ generate_advice → ⑬ format_response → END

注:④ extract_symptoms 节点已删除 — ④ 重设计为 1 LLM 直接从 state 选追问
(空 slot 维度填补 + 开放式症状兜底),不再依赖 ④ 抽出来的关键词。

`build_graph()` 返回未编译的 StateGraph(便于测试时注入 checkpointer);
`build_app()` 返回编译后的可执行 app(默认 InMemorySaver,interrupt 需要 checkpointer)。
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.agent.nodes.analyze_initial_reports import analyze_initial_reports
from src.agent.nodes.build_query import build_query
from src.agent.nodes.diagnose import diagnose
from src.agent.nodes.format_response import format_response
from src.agent.nodes.generate_advice import generate_advice
from src.agent.nodes.generate_followup import generate_followup
from src.agent.nodes.info_collect import info_collect
from src.agent.nodes.process_exam_result import process_exam_result
from src.agent.nodes.process_followup import process_followup_answer
from src.agent.nodes.recommend_exam import recommend_exam
from src.agent.nodes.retrieve import retrieve
from src.agent.nodes.safety_gate import safety_gate
from src.agent.nodes.select_symptom import select_discriminative_symptom
from src.agent.nodes.wait_exam_report import wait_exam_report
from src.agent.nodes.wait_followup_answer import wait_followup_answer
from src.agent.routers.diagnose_router import diagnose_router
from src.agent.routers.should_continue import should_continue
from src.agent.state import MedicalState


def build_graph() -> StateGraph:
    """构造未编译的 StateGraph(测试 / 调试用)。"""
    workflow = StateGraph(MedicalState)

    # 15 节点(原 16 节点,④ extract_symptoms 已删)
    workflow.add_node("info_collect", info_collect)
    workflow.add_node("analyze_initial_reports", analyze_initial_reports)
    workflow.add_node("build_query", build_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("select_discriminative_symptom", select_discriminative_symptom)
    workflow.add_node("generate_followup", generate_followup)
    workflow.add_node("wait_followup_answer", wait_followup_answer)
    workflow.add_node("process_followup_answer", process_followup_answer)
    workflow.add_node("recommend_exam", recommend_exam)
    workflow.add_node("wait_exam_report", wait_exam_report)
    workflow.add_node("process_exam_result", process_exam_result)
    workflow.add_node("diagnose", diagnose)
    workflow.add_node("safety_gate", safety_gate)
    workflow.add_node("generate_advice", generate_advice)
    workflow.add_node("format_response", format_response)

    # 入口
    workflow.set_entry_point("info_collect")

    # 顺序边 ①→①.5→②→③→④(④ 已删)
    workflow.add_edge("info_collect", "analyze_initial_reports")
    workflow.add_edge("analyze_initial_reports", "build_query")
    workflow.add_edge("build_query", "retrieve")
    workflow.add_edge("retrieve", "select_discriminative_symptom")

    # 条件边:④ → 追问 / 诊断
    workflow.add_conditional_edges(
        "select_discriminative_symptom",
        should_continue,
        {
            "followup": "generate_followup",
            "diagnose": "diagnose",
        },
    )

    # 追问循环:⑤ → ⑥ → ⑦ → ②
    workflow.add_edge("generate_followup", "wait_followup_answer")
    workflow.add_edge("wait_followup_answer", "process_followup_answer")
    workflow.add_edge("process_followup_answer", "build_query")

    # 检查循环:⑧a → ⑧b → ⑨ → ②
    workflow.add_edge("recommend_exam", "wait_exam_report")
    workflow.add_edge("wait_exam_report", "process_exam_result")
    workflow.add_edge("process_exam_result", "build_query")

    # 条件边:⑩ → 检查 / 安全门控
    workflow.add_conditional_edges(
        "diagnose",
        diagnose_router,
        {
            "recommend_exam": "recommend_exam",
            "safety_gate": "safety_gate",
        },
    )

    # 安全门控 → 建议 → 输出 → END
    workflow.add_edge("safety_gate", "generate_advice")
    workflow.add_edge("generate_advice", "format_response")
    workflow.add_edge("format_response", END)

    return workflow


def build_app(checkpointer=None):
    """编译可执行 app。interrupt 节点需要 checkpointer,默认 InMemorySaver。

    生产环境可注入 PostgresSaver / RedisSaver 持久化中断点。
    """
    return build_graph().compile(checkpointer=checkpointer or InMemorySaver())
