"""src/agent/nodes/format_response.py — Agent ⑬ format_response(DEV_SPEC §4.1.2 ⑬)。

LLM 自由文本(无 schema)— 把诊断 + 建议 + 风险提示 + 免责声明组成患者可读回复。
失败兜底:返回**模板化的简短免责声明**,保证用户侧始终有可执行落点。
"""
from __future__ import annotations

import logging
import time

from src.agent.state import MedicalState
from src.common.metrics import (
    _attempts,
    _failures,
    _fallbacks,
    _latency,
    retry_observer,
)
from config.settings import settings
from src.models.llm_client import get_llm
from src.prompts.agent import build_format_response_prompt


_logger = logging.getLogger(__name__)
_NODE = "format_response"
_SCHEMA = "free_text"

_FALLBACK_RESPONSE = (
    "本次咨询暂未生成完整回复,请稍后重试或线下就诊。"
    "如有不适紧急加重,请立即前往最近的医疗机构就诊。"
)


def format_response(state: MedicalState) -> dict:
    failure_reason = None
    if state.diagnosis_result:
        failure_reason = state.diagnosis_result[0].get("failure_reason")

    prompt = build_format_response_prompt(
        diagnosis_results=state.diagnosis_result,
        medication_advice=state.medication_advice,
        recommended_tests=state.recommended_tests,
        risk_warnings=state.risk_warnings,
        failure_reason=failure_reason,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        # 2026-05-24:⑬ 是纯 reformatting(把 state 已有结构化结果改写成患者段落),
        # 不涉及新临床判断,切 flash 省 5-10s(评测显示 pro 在此任务无收益)
        chain = get_llm(model=settings.llm.FAST_MODEL_NAME).with_retry(stop_after_attempt=3)
        msg = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA},
            },
        )
        text = (msg.content if hasattr(msg, "content") else str(msg)).strip()
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _fallbacks.labels(node=_NODE, fallback_type="static_template").inc()
        _logger.error("[%s] free-text LLM call failed, returning template: %s", _NODE, e)
        text = _FALLBACK_RESPONSE
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", _NODE, elapsed)

    return {"final_response": text}
