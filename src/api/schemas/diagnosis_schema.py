"""问诊接口 Pydantic 模型(DEV_SPEC §8.4 G4 / §9.6)。

请求 / 响应通用约定:
- `session_id` 首次为空,后端建 sessions 行回传;后续轮次客户端必须 echo 同一个
- `status` 是核心状态机:
    "ongoing_followup"   → 服务端发起追问,前端拿 `pending_question` 提示用户
    "ongoing_exam"       → 服务端要求线下检查,前端拿 `recommended_tests` 引导
    "completed"          → 终态,带完整 `diagnosis_result` + `medication_advice` 等
- 状态机由 LangGraph interrupt() 自然驱动 — 任何 interrupt 触发即返 ongoing_*,
  graph 跑完才 completed
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    """POST /diagnose 请求体。

    首次问诊:`session_id=None, patient_input="<主诉>"`
    回答追问:`session_id=<上次返回的>, followup_answer="<回答>"`
    回传检查报告:`session_id=<...>, exam_results=[{"file_ref": "<路径>"}, ...]`
    """

    session_id: str | None = Field(None, description="首次为空,后端建 sessions 行后回传")
    patient_input: str | None = Field(
        None,
        description="主诉(首次必填;追问/检查回传时不需要,会被忽略)",
        max_length=4000,
    )
    followup_answer: str | None = Field(
        None, description="对上一轮 pending_question 的回答(追问轮使用)"
    )
    exam_results: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "线下检查报告回传,2026-05-22 改为**按 group 分组**格式:"
            "每项 `{group_label: str, files: list[str], status: 'uploaded'|'skipped'}`,"
            "对应 ⑧a 推荐的每组。files 是 /diagnose/upload 落盘后返回的 file_ref 路径。"
            "status='skipped' 表示患者未做该组,⑨ 跳过解析。"
        ),
    )


DiagnoseStatus = Literal["ongoing_followup", "ongoing_exam", "completed"]


class DiagnoseResponse(BaseModel):
    """POST /diagnose 响应体。字段语义与 graph 状态机对齐(spec §4.1.2 ⑤+⑥/⑧/⑬)。"""

    session_id: str
    status: DiagnoseStatus

    pending_question: str | None = Field(
        None, description="待用户回答的追问(完整文本)。status='ongoing_followup' 时非空"
    )
    pending_questions: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "结构化追问条目(state.followup_questions 镜像),供前端按 type 渲染输入控件。"
            "例如 type=obstetric 渲染 yes/no 按钮,type=slot/open/history/targeted 渲染文本框。"
            "status='ongoing_followup' 时非空,前端可选用;后端 ⑦ 仍按 followup_answer 文本解析。"
        ),
    )
    recommended_test_groups: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "2026-05-22:推荐检查的**分组**结构(替代旧 recommended_tests)。"
            "status='ongoing_exam' 时非空,前端按 group 渲染独立上传框。"
            "每项 dict 形态 {group_label: str, items: list[str], note: str},"
            "见 src/agent/schemas/recommend_exam.py:ExamGroup。"
        ),
    )
    recommended_tests: list[str] | None = Field(
        None,
        description=(
            "扁平自然语言检查建议(由 ⑫ generate_advice 写,⑬ format_response 拼最终回复用)。"
            "在 completed 状态返回。**不要**用于 ongoing_exam 引导用户上传 — 那是 "
            "recommended_test_groups 的职责。"
        ),
    )

    final_response: str | None = Field(
        None, description="完整诊断回复(给患者展示的最终文字)"
    )
    diagnosis_result: list[dict[str, Any]] | None = Field(
        None, description="结构化诊断结果列表(每项含 disease/probability/evidence/differentiation/differentiation_type/failure_reason)"
    )
    medication_advice: list[dict[str, Any]] | None = Field(
        None, description="用药建议列表"
    )
    risk_warnings: list[str] | None = Field(
        None, description="风险提示(safety_gate ⑪ 兜底 + generate_advice ⑫ 系统提示)"
    )
