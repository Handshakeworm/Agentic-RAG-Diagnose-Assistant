"""src/agent/nodes/analyze_initial_reports.py — Agent ①.5 节点(DEV_SPEC §4.1.2 ①.5)。

节点入口 `interrupt(...)` 询问患者是否要上传近期检查报告,resume 后:
1. 用户上传(模拟)→ 给 state.exam_reports 加 mock 占位 + report_findings 占位(demo 不真跑 VLM)
2. 用户跳过 → 仍按原逻辑从 PG 拉历史 exam_reports,有就 VLM 真解析,无则透传

报告上传通路当前为前端模拟(`ui/index.html` 文件选择不发后端);demo 演示交互链路完整。
真接 multipart 上传 + 持久化是单独任务,本节点占位 placeholder 不影响后续 ② build_query。

复用边界:VLM 解析 `report_parser.parse_reports()` 跟 ⑨ process_exam_result 共享。
"""
from __future__ import annotations

from langgraph.types import interrupt

from src.agent.state import MedicalState
from src.agent.utils.patient_repo import load_initial_exam_reports
from src.agent.utils.report_parser import parse_reports


# interrupt() 第一次 raise 时的 payload(给 SSE 层走 snapshot.tasks 取用);
# 文案放这里方便单点改。
INITIAL_REPORT_UPLOAD_PROMPT = {
    "type": "report_upload",
    "question": "您有近期的检查报告(化验单、影像、心电图等)要一起分析吗?可选,没有就跳过。",
}


def analyze_initial_reports(state: MedicalState) -> dict:
    """interrupt 询问上传 → resume 后接收用户回答 → 加载/分析 exam_reports。

    LangGraph interrupt 语义:第一次执行到 `interrupt(payload)` 抛 GraphInterrupt,
    client resume 时 interrupt(...) 直接返回 client 传入的 resume value。本节点 body
    会被整段重跑,但 interrupt 调用变成直接 return,继续下游逻辑。
    """
    upload_response = interrupt(INITIAL_REPORT_UPLOAD_PROMPT)

    # demo 模拟通路:upload_response 是 ⑦ 序列化前端 form 回来的字符串,含"已上传"
    # 表示用户选了文件(前端只发文件名,不发字节);此时给 state 加 placeholder,
    # **跳过真 VLM 调用**(demo 不真发后端解析,避免误算 token / 误信结果)。
    mock_uploaded = isinstance(upload_response, str) and "已上传" in upload_response

    # 始终 load PG 已有(可能为空);跟 mock 上传是两个独立来源
    exam_reports = load_initial_exam_reports(state.patient_id)
    update: dict = {"exam_reports": exam_reports}

    if mock_uploaded:
        update["report_findings"] = [{
            "name": "用户上传的报告(demo placeholder)",
            "summary": (
                "本 demo 仅展示上传交互链路,未做实际多模态 VLM 解析。"
                f"用户回答原文:{upload_response}"
            ),
        }]
        return update

    if not exam_reports:
        return update

    file_refs = [r["file_ref"] for r in exam_reports if r.get("file_ref")]
    findings = parse_reports(file_refs)
    update["report_findings"] = findings
    return update
