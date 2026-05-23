"""src/agent/nodes/wait_exam_report.py — Agent ⑧b wait_exam_report(DEV_SPEC §4.1.5)。

interrupt 等待患者线下就医后回传检查结果(图片/PDF);恢复时只重执行本节点,
⑧a 已生成的 recommended_test_groups 不会重复 LLM 调用。

2026-05-22 改造:interrupt payload 从扁平 `recommended_tests: list[str]` 改成
`recommended_test_groups: list[dict]`(分组结构),前端按 group 出独立上传框,
每组允许 0~N 个文件 + skip 标记。

interrupt 返回值:list[dict],每项 `{group_label, files: list[str], status}`,
  - files: 落盘后的文件路径(由前端 /diagnose/upload endpoint 落盘后返回)
  - status: "uploaded" / "skipped"(患者点跳过)/ "pending"(预留)
"""
from __future__ import annotations

from langgraph.types import interrupt

from src.agent.state import MedicalState


def wait_exam_report(state: MedicalState) -> dict:
    """暂停执行,interrupt 返回值写入 pending_exam_results。"""
    pending = interrupt(state.recommended_test_groups)
    return {"pending_exam_results": pending or []}
