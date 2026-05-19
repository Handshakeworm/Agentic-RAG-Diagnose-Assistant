"""src/agent/nodes/generate_followup.py — Agent ⑤ generate_followup(DEV_SPEC §4.1.2 ⑤+⑥)。

LLM 把 followup_questions(混合维度+症状)合并成 2-3 句口语化追问。
**自由文本输出**(无 schema)— 按 §9.4 / §9.1 模板裸写,schema 标签固定 "free_text"。
拆分点:LLM 调用与 interrupt 等待分属两个节点(⑤ / ⑥),避免 interrupt 恢复时
重复调 LLM(spec §4.1.2 ⑤+⑥ 拆分设计注)。
"""
from __future__ import annotations

import logging
import time

from src.agent.state import MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_followup_question_prompt


_logger = logging.getLogger(__name__)
_NODE = "generate_followup"
_SCHEMA = "free_text"


def generate_followup(state: MedicalState) -> dict:
    """生成自然语言追问问题,写入 followup_question。"""
    if not state.followup_questions:
        # 路由层已保证非空,这里保险兜底,空时直接透传
        return {"followup_question": ""}

    prompt = build_followup_question_prompt(
        chief_complaint=state.chief_complaint,
        questions=state.followup_questions,
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        llm = get_llm()
        # 自由文本无需 with_structured_output;with_retry 仍按中级
        chain = llm.with_retry(stop_after_attempt=3)
        msg = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA},
            },
        )
        # ChatOpenAI 返回 AIMessage,取 .content
        question = (
            msg.content if hasattr(msg, "content") else str(msg)
        ).strip()
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _logger.error("[%s] free-text LLM call failed: %s", _NODE, e)
        raise
    finally:
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(
            time.perf_counter() - t0
        )

    return {"followup_question": question}
