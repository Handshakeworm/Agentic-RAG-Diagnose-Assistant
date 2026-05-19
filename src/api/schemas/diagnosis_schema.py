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
            "线下检查报告回传,每项 `{file_ref, ...}`(检查回传轮使用)。"
            "file_ref 是已上传文件的路径或 URL"
        ),
    )


DiagnoseStatus = Literal["ongoing_followup", "ongoing_exam", "completed"]


class DiagnoseResponse(BaseModel):
    """POST /diagnose 响应体。字段语义与 graph 状态机对齐(spec §4.1.2 ⑥/⑧/⑬)。"""

    session_id: str
    status: DiagnoseStatus

    pending_question: str | None = Field(
        None, description="待用户回答的追问。status='ongoing_followup' 时非空"
    )
    recommended_tests: list[str] | None = Field(
        None,
        description=(
            "建议的线下检查项目列表。status='ongoing_exam' 时非空,"
            "前端引导用户去医院做检查后调本接口回传 exam_results"
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
