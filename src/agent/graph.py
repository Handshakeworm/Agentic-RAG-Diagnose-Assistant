"""src/agent/graph.py — Agent LangGraph StateGraph 编排(DEV_SPEC §4.1.4)。

注册 17 节点 + 3 条件边。**form 体验设计**:⓪a / ①.5 / intake / ⑥ 都有 interrupt:

  START → ⓪a initial_ask(interrupt 弹 form-1: open/病史/孕期) →
  ① info_collect(LLM 同时抽主诉 + 解析 ⓪a form) →
  ①.5 analyze_initial_reports(interrupt 弹 form-2: 报告) →
  intake_followup_ask(**multi-interrupt 收 13 维 slot batch + LLM holistic 翻译**) →
  ⑤ generate_followup
    ┌─[generate_followup_out_router 路由]─┐
    │  to_wait → ⑥(interrupt 弹 form-targeted) → ⑦
    │             ┌─[post_followup_router 路由]─┐
    │             │  loop_to_followup(source=intake) → 回 ⑤(LLM 再判 targeted)
    │             └─ to_build_query(source=diagnostic 或 round 触顶)→ ②
    └─ to_build_query(intake 阶段 LLM 判够了 / 无追问) → ② build_query →
       ③ retrieve → ④ select_discriminative_symptom →
       ┌─[should_continue 路由]─┐
       │  followup(④ 写 followup_source="diagnostic") → ⑤ → ⑥ → ⑦ → ②(回检索 → ④ 再判)
       └─ diagnose → ⑩ diagnose →
          ┌─[diagnose_router 路由]─┐
          │  recommend_exam → ⑧a → ⑧b(interrupt) → ⑨ → ②
          └─ safety_gate → ⑪ → ⑫ → ⑬ → END

注:
- ⓪a → ① 串行直连(① 单次 LLM 同时抽主诉/HPI + 解析 ⓪a form 答案)
- **intake_followup_ask 一次性重型节点**:节点内 multi-interrupt 收 13 维 slot batch
  (每批 4+open),全部收完一次 LLM holistic 翻译(复用 ⑦ 的 parse_followup_response
  helper),写 present_illness_slots / medical_history / confirmed_symptoms + 设
  `followup_source="intake"`。**不**出 targeted 题(那归 ⑤),**不**被 loop(出了就不回)
- **⑤ generate_followup 双职责**:
  (a) followup_questions 已被人写(④ 鉴别诊断)→ LLM 生成口语化文案
  (b) followup_questions 空 + source==intake → LLM 看 holistic 决定 targeted 追问
      (复用原 intake 末尾的 _llm_targeted_phase 逻辑)
- **⑦ 单一职责**:LLM 翻译 ⑥ 后的追问回答,清掉 followup_questions
- **post_followup_router**(⑦ 后)2 路:source=intake → 回 ⑤ loop(让 ⑤ LLM 再判);
  source=diagnostic → ②(④ 鉴别诊断追问完成后 ⑩ 重新走)
- 兜底:`followup_round >= MAX_FOLLOWUP_ROUNDS=8` 强制走 ② 防 LLM 死循环

`build_graph()` 返回未编译的 StateGraph(便于测试时注入 checkpointer);
`build_app()` 返回编译后的可执行 app(默认 InMemorySaver,interrupt 需要 checkpointer)。
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes.analyze_initial_reports import analyze_initial_reports
from src.agent.nodes.build_query import build_query
from src.agent.nodes.diagnose import diagnose
from src.agent.nodes.format_response import format_response
from src.agent.nodes.generate_advice import generate_advice
from src.agent.nodes.generate_followup import generate_followup
from src.agent.nodes.info_collect import info_collect
from src.agent.nodes.initial_ask import initial_ask
from src.agent.nodes.intake_followup_ask import intake_followup_ask
from src.agent.nodes.process_exam_result import process_exam_result
from src.agent.nodes.process_followup import process_followup_answer
from src.agent.nodes.recommend_exam import recommend_exam
from src.agent.nodes.retrieve import retrieve
from src.agent.nodes.safety_gate import safety_gate
from src.agent.nodes.select_symptom import select_discriminative_symptom
from src.agent.nodes.wait_exam_report import wait_exam_report
from src.agent.nodes.wait_followup_answer import wait_followup_answer
from src.agent.routers.diagnose_router import diagnose_router
from src.agent.routers.intake_router import (
    generate_followup_out_router,
    post_followup_router,
)
from src.agent.routers.should_continue import should_continue
from src.agent.state import MedicalState


def build_graph() -> StateGraph:
    """构造未编译的 StateGraph(测试 / 调试用)。"""
    workflow = StateGraph(MedicalState)

    # 17 节点(④ extract_symptoms 已删;新增 ⓪a initial_ask 与 ① 并行,
    # 新增 intake_followup_ask 在 ①.5 与 ② 之间承担入站追问)
    workflow.add_node("initial_ask", initial_ask)
    workflow.add_node("info_collect", info_collect)
    workflow.add_node("analyze_initial_reports", analyze_initial_reports)
    workflow.add_node("intake_followup_ask", intake_followup_ask)
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

    # 入口:START → ⓪a(单边)。⓪a 节点入口 interrupt 弹综合 form(open/病史/孕期),
    # 用户答完直接进 ①(① 一次 LLM 同时抽主诉/HPI + 解析 ⓪a form 答案)。
    workflow.add_edge(START, "initial_ask")

    # ⓪a → ① → ①.5(interrupt 问报告)→ intake_followup_ask → ⑤ → ⑥ → ⑦
    workflow.add_edge("initial_ask", "info_collect")
    workflow.add_edge("info_collect", "analyze_initial_reports")
    workflow.add_edge("analyze_initial_reports", "intake_followup_ask")
    workflow.add_edge("intake_followup_ask", "generate_followup")

    # 检索主链路
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

    # ⑤ 出口:有题 → ⑥;无题(intake 阶段 LLM 判够了)→ ②
    workflow.add_conditional_edges(
        "generate_followup",
        generate_followup_out_router,
        {
            "to_wait": "wait_followup_answer",
            "to_build_query": "build_query",
        },
    )

    # ⑥ → ⑦ → post_followup_router 分流:
    #   intake 阶段(followup_source=intake)→ 回 ⑤ 让 LLM 再判 targeted
    #   diagnostic 阶段(④ 写的 source=diagnostic)→ ② 回检索 → ④ 再判
    workflow.add_edge("wait_followup_answer", "process_followup_answer")
    workflow.add_conditional_edges(
        "process_followup_answer",
        post_followup_router,
        {
            "loop_to_followup": "generate_followup",
            "to_build_query": "build_query",
        },
    )

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
