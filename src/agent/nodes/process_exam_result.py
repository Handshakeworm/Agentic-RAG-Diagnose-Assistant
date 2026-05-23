"""src/agent/nodes/process_exam_result.py — Agent ⑨ process_exam_result(DEV_SPEC §4.1.2 ⑨)。

消费 ⑧b 写入的 pending_exam_results(2026-05-22 起按 group 分组):
1. 遍历每个 group:
   - status == "skipped" → 跳过,记 log(patient 未做该项;⑩ 看到无此组报告会推断为缺失)
   - status == "uploaded" → 把 group.files 传给 parse_reports,带 hint = group_label + items
                            (帮多模态 LLM 定位报告类型,提升解析准确度)
2. 文件 ref 追加到 exam_reports(每项带 group_label 元数据帮 ⑩ 追溯)
3. parse_reports 产 ReportFinding 追加到 report_findings(也带 group_label)
4. 流程回到 build_query,带新证据重新召回

注:文件落盘步骤由前端/API 层在 /diagnose/upload endpoint 完成,本节点假设
pending_exam_results 的 files 已是落盘后的路径。
"""
from __future__ import annotations

import logging

from src.agent.state import MedicalState
from src.agent.utils.report_parser import parse_reports


_logger = logging.getLogger(__name__)


def process_exam_result(state: MedicalState) -> dict:
    pending = state.pending_exam_results or []
    if not pending:
        return {}

    new_refs: list[dict] = []
    new_findings: list[dict] = []
    skipped_labels: list[str] = []

    for group in pending:
        if not isinstance(group, dict):
            continue
        label = (group.get("group_label") or "").strip()
        status = group.get("status") or "uploaded"
        files = list(group.get("files") or [])

        if status == "skipped" or not files:
            if label:
                skipped_labels.append(label)
                _logger.info(
                    "[process_exam_result] group=%r skipped(无文件或患者跳过)", label
                )
            continue

        # 组装 hint:label + 期望含的 items + 原 note(从 recommended_test_groups 找)
        hint = _build_hint(state.recommended_test_groups, label)

        # files 追加 exam_reports(每项带 group_label 元数据,便于 ⑩ 追溯)
        for fref in files:
            new_refs.append({"file_ref": fref, "group_label": label})

        # parse_reports 按本组的 files 调一次多模态(带 hint)
        group_findings = parse_reports(files, hint=hint)
        for f in group_findings:
            f.setdefault("group_label", label)  # finding 也带 group_label 帮追溯
        new_findings.extend(group_findings)

    if skipped_labels:
        _logger.info(
            "[process_exam_result] 本批跳过组数=%d: %s",
            len(skipped_labels), skipped_labels,
        )

    # 追加到 exam_reports / report_findings,report_index 在 parse_reports 中已补
    # 但 parse_reports 给的 index 是基于本批 file_paths(0..N-1),需要重映射到全局
    base = len(state.exam_reports)
    for i, f in enumerate(new_findings):
        f["report_index"] = base + i

    return {
        "exam_reports": list(state.exam_reports) + new_refs,
        "report_findings": list(state.report_findings) + new_findings,
        # 清空 pending 防重复消费
        "pending_exam_results": [],
    }


def _build_hint(test_groups: list[dict], label: str) -> str | None:
    """从 recommended_test_groups 找匹配 group 的 items + note,组装 hint 文案。

    返回 None 表示找不到匹配(label 漂或者 ⑧a 没出过这组)→ parse_reports 不带 hint。
    """
    if not label:
        return None
    for g in test_groups or []:
        if (g.get("group_label") or "").strip() == label:
            items = ", ".join(g.get("items") or [])
            note = (g.get("note") or "").strip()
            hint = f"这组报告是 '{label}',期望包含的检查项: {items}。"
            if note:
                hint += f" 提示: {note}"
            return hint
    return None
