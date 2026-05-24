"""src/agent/nodes/analyze_initial_reports.py — Agent ①.5 节点(DEV_SPEC §4.1.2 ①.5)。

节点入口 `interrupt(...)` 询问患者是否要上传近期检查报告,resume 后:
1. 上传路径(2026-05-23 起,resume value 是 list[group dict],跟 ⑧b/⑨ 共结构):
   - 每个 group {group_label, files, status}
   - 遍历非 skipped 且 files 非空的组 → parse_reports(hint=group_label)→ exam_reports + report_findings
   - 不再混入 PG 历史(用户主动传了新报告就只看新)
2. 跳过路径(resume value 是空 list / 空 str / 其他兼容降级):
   - 仍尝试 load_initial_exam_reports(state.patient_id):老患者上次诊断留的报告
   - 有就 VLM 真解析,无则透传空

`exam_results` 跟 ⑧b 共享 SSE resume 通路(同 DiagnoseRequest.exam_results 字段);
LangGraph 按 snapshot.next 决定 resume value 喂给哪个节点,所以两者复用同一字段不冲突。

复用边界:VLM 解析 `report_parser.parse_reports()` 跟 ⑨ process_exam_result 共享。
"""
from __future__ import annotations

import logging

from langgraph.types import interrupt

from src.agent.state import MedicalState
from src.agent.utils.patient_repo import load_initial_exam_reports
from src.agent.utils.report_parser import parse_reports, parse_reports_parallel


_logger = logging.getLogger(__name__)


# interrupt() 第一次 raise 时的 payload(给 SSE 层走 snapshot.tasks 取用);
# 文案放这里方便单点改。
INITIAL_REPORT_UPLOAD_PROMPT = {
    "type": "report_upload",
    "question": "您有近期的检查报告(化验单、影像、心电图等)要一起分析吗?可选,没有就跳过。",
}


def analyze_initial_reports(state: MedicalState) -> dict:
    """interrupt 询问上传 → resume 后接收 list[group dict] → parse_reports 真解析。

    LangGraph interrupt 语义:第一次执行到 `interrupt(payload)` 抛 GraphInterrupt,
    client resume 时 interrupt(...) 直接返回 client 传入的 resume value。本节点 body
    会被整段重跑,但 interrupt 调用变成直接 return,继续下游逻辑。
    """
    upload_response = interrupt(INITIAL_REPORT_UPLOAD_PROMPT)

    # 2026-05-23:新契约 resume value 是 list[{group_label, files, status}],
    # 跟 ⑧b/⑨ 共享结构。空 list / 非 list 类型(老 str / None)→ 走 PG fallback。
    groups: list[dict] = []
    if isinstance(upload_response, list):
        groups = [g for g in upload_response if isinstance(g, dict)]

    if not groups:
        return _pg_fallback(state)

    return _process_uploaded_groups(groups)


def _pg_fallback(state: MedicalState) -> dict:
    """跳过路径:load PG 历史 exam_reports,有就 VLM 解析,无则空透传。"""
    exam_reports = load_initial_exam_reports(state.patient_id)
    update: dict = {"exam_reports": exam_reports}
    if not exam_reports:
        return update

    file_refs = [r["file_ref"] for r in exam_reports if r.get("file_ref")]
    findings = parse_reports(file_refs)
    update["report_findings"] = findings
    return update


def _process_uploaded_groups(groups: list[dict]) -> dict:
    """上传路径:遍历每组 → **并行**调 parse_reports(hint=group_label) → 组装 exam_reports + findings。

    首诊场景 base index = 0(首次写 state.exam_reports);group_label 帮 ⑩ 追溯。
    多组 VLM 解析走 `parse_reports_parallel`(线程池并发,典型 3 组省 50%+ 延迟)。
    """
    new_refs: list[dict] = []
    new_findings: list[dict] = []
    skipped_labels: list[str] = []
    tasks: list[tuple[list[str], str | None]] = []
    task_labels: list[str] = []  # 跟 tasks 同序,回填 group_label

    for group in groups:
        label = (group.get("group_label") or "").strip()
        status = group.get("status") or "uploaded"
        files = list(group.get("files") or [])

        if status == "skipped" or not files:
            if label:
                skipped_labels.append(label)
                _logger.info(
                    "[analyze_initial_reports] group=%r skipped(无文件或患者跳过)", label
                )
            continue

        # hint:首诊只有 group_label(没有 recommended items / note),hint 简化
        hint = f"这份报告标签: {label}" if label else None

        for fref in files:
            entry: dict = {"file_ref": fref}
            if label:
                entry["group_label"] = label
            new_refs.append(entry)

        tasks.append((files, hint))
        task_labels.append(label)

    # 并行 VLM 解析(每 task 独立调用,无共享状态)
    findings_per_task = parse_reports_parallel(tasks)
    for label, group_findings in zip(task_labels, findings_per_task):
        for f in group_findings:
            if label:
                f.setdefault("group_label", label)
        new_findings.extend(group_findings)

    if skipped_labels:
        _logger.info(
            "[analyze_initial_reports] 本批跳过组数=%d: %s",
            len(skipped_labels), skipped_labels,
        )

    # report_index 重映射:首诊 base = 0
    for i, f in enumerate(new_findings):
        f["report_index"] = i

    return {
        "exam_reports": new_refs,
        "report_findings": new_findings,
    }
