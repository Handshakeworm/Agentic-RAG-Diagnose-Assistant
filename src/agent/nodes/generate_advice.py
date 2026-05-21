"""src/agent/nodes/generate_advice.py — Agent ⑫ generate_advice(DEV_SPEC §4.1.2 ⑫)。

读 diagnosis_result[0].failure_reason,在 risk_warnings 追加系统级提示:
  - None                                  → 正常推理结果(按 confirmed/insufficient/
                                            need_exam at cap 分支处理)
  - "followup_round_capped"               → 追加 "本次问诊轮次较多仍未收敛..."
  - "step_N_structured_output_failed: ..."→ 追加 "系统分析过程出现技术问题..."
                                            (不暴露异常细节)

LLM 给结构化建议(中安全等级,失败抛异常)。
"""
from __future__ import annotations

import logging
import time

from src.agent.schemas.advice import AdviceOutput
from src.agent.state import MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_advice_prompt


_logger = logging.getLogger(__name__)
_NODE = "generate_advice"
_SCHEMA = "AdviceOutput"

_FAILURE_NOTE_CAPPED = "本次问诊轮次较多仍未收敛,建议线下就诊以获得更全面的评估"
_FAILURE_NOTE_STEP = "系统分析过程出现技术问题,本次诊断结果不可作为依据,请尽快线下就诊"

_SEVERITY_LABEL: dict[str, str] = {
    "high": "高风险",
    "medium": "中风险",
    "low": "低风险",
}


def _system_failure_note(failure_reason: str | None) -> str | None:
    if failure_reason is None:
        return None
    if failure_reason == "followup_round_capped":
        return _FAILURE_NOTE_CAPPED
    if failure_reason.startswith("step_"):
        return _FAILURE_NOTE_STEP
    return None


def _format_additional_risk(item: dict) -> str:
    """⑪ safety_gate LLM 兜底产出的 additional_risk 项 → 患者可读单行警告。

    item schema 见 spec §9.5 SafetyGateOutput.additional_risks 子项:
        {risk_type, description, severity, recommendation}
    """
    parts: list[str] = []
    severity_label = _SEVERITY_LABEL.get(item.get("severity") or "")
    if severity_label:
        parts.append(f"[{severity_label}]")
    desc = (item.get("description") or "").strip()
    if desc:
        parts.append(desc)
    rec = (item.get("recommendation") or "").strip()
    if rec:
        parts.append(f"建议:{rec}")
    return " ".join(parts).strip()


def generate_advice(state: MedicalState) -> dict:
    failure_reason = None
    if state.diagnosis_result:
        failure_reason = state.diagnosis_result[0].get("failure_reason")

    prompt = build_advice_prompt(
        diagnosis_results=state.diagnosis_result,
        safety_constraints=state.safety_constraints,
        failure_reason=failure_reason,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(AdviceOutput, method="json_mode").with_retry(stop_after_attempt=3)
        result: AdviceOutput = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA},
            },
        )
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _logger.error("[%s] structured output failed: %s", _NODE, e, exc_info=True)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", _NODE, elapsed)

    risk_warnings = list(result.risk_warnings)

    # spec §4.1.2 ⑪→⑫ 接缝:消费 ⑪ LLM 兜底产出的 additional_risks(交叉过敏 /
    # 妊娠禁忌 / 剂量调整等),格式化后并入 risk_warnings,避免 LLM 兜底白调
    for item in (state.safety_constraints or {}).get("additional_risks") or []:
        text = _format_additional_risk(item)
        if text and text not in risk_warnings:
            risk_warnings.append(text)

    note = _system_failure_note(failure_reason)
    if note and note not in risk_warnings:
        risk_warnings.append(note)

    return {
        "medication_advice": [m.model_dump() for m in result.medications],
        "recommended_tests": result.exam_suggestions or list(state.recommended_tests),
        "risk_warnings": risk_warnings,
    }
