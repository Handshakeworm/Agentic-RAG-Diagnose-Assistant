"""src/agent/nodes/info_collect.py — Agent ① info_collect 节点(DEV_SPEC §4.1.2 ①)。

单次 LLM 同时处理两路输入,合并了 ⑦ 在原 to_info_collect 路径上的活:
- 从 `patient_input` 抽 chief_complaint / present_illness / 13 维 slots
- 从 ⓪a 综合 form 答案抽 history_fills / obstetric_fills / new_symptoms

节点 return 时把 history/obstetric merge 进 state.medical_history(复用
`process_followup` 同款 helper),new_symptoms append 进 confirmed_symptoms。
medical_history 的初始档案由 ⓪a load,本节点只增量合并,**不再整段写**(否则丢 ⓪a 的 load)。

为什么合并 ⓪a-form 处理:⑦ 在 ⑥ 之后服务"slot/targeted 追问解析",跟 ⓪a 综合 form 解析
语义同质但来源不同;首轮 to_info_collect 路径上"⓪a → ⑦ → ①"两次 LLM 调用浪费 ~3s,
现在合并到 ① 一次完成。⑦ 后续只服务 ⑥ 后的追问解析,职责单一。

**模型选择**:本节点用 `settings.llm.FAST_MODEL_NAME`(deepseek-v4-flash),不是主链路
默认的 -pro。原因:① 是首交互瓶颈,直接影响用户看到下一个 form 的延迟。HPI 抽取任务
对推理深度要求低(就是 ~500 tokens 的结构化转写),flash 比 pro 快 2~3x,精度差异
可忽略。其余主链路节点继续走 MODEL_NAME。

LLM 调用按 §9.1 模板裸写 try/except/finally,6 指标手动上报。
"""
from __future__ import annotations

import logging
import time

from config.settings import settings
from src.agent.nodes.process_followup import (
    _merge_history_fills,
    _merge_obstetric_fills,
)
from src.agent.schemas.info_collect import InfoCollectOutput
from src.agent.state import MedicalState, PresentIllnessSlots
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_info_collect_prompt


_logger = logging.getLogger(__name__)

_NODE = "info_collect_step1"
_SCHEMA = "InfoCollectOutput"


def info_collect(state: MedicalState) -> dict:
    """LLM 提取主诉/HPI + 解析 ⓪a form 答案,return state 更新 dict。"""
    prompt = build_info_collect_prompt(
        patient_input=state.patient_input,
        initial_form_question=state.followup_question or "",
        initial_form_answer=state.followup_answer or "",
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm(model=settings.llm.FAST_MODEL_NAME).with_structured_output(InfoCollectOutput, method="json_mode").with_retry(stop_after_attempt=3)
        result: InfoCollectOutput = chain.invoke(
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
        raise  # 中安全:抛回 graph,终止会话(无主诉无法继续)
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", _NODE, elapsed)

    # ─── 把 ⓪a form 答案的解析结果 merge 进 state(复用 ⑦ 的 helper)───
    new_medical_history = _merge_obstetric_fills(state.medical_history, result.obstetric_fills)
    new_medical_history = _merge_history_fills(new_medical_history, result.history_fills)

    confirmed = list(state.confirmed_symptoms)
    denied = list(state.denied_symptoms)
    uncertain = list(state.uncertain_symptoms)

    # 症状三分类合并:LLM 按语气把 ⓪a form open 题 + patient_input 提及的症状
    # 分到 confirmed/denied/uncertain,供下游 ⑤ 去重避免重复追问已知症状。
    already_known = set(confirmed) | set(denied) | set(uncertain)
    for term in (result.confirmed_symptoms or []):
        if term and term not in already_known:
            confirmed.append(term)
            already_known.add(term)
    for term in (result.denied_symptoms or []):
        if term and term not in already_known:
            denied.append(term)
            already_known.add(term)
    for term in (result.uncertain_symptoms or []):
        if term and term not in already_known:
            uncertain.append(term)
            already_known.add(term)

    return {
        "chief_complaint": result.chief_complaint,
        "present_illness": result.present_illness,
        "present_illness_slots": PresentIllnessSlots(
            **result.present_illness_slots.model_dump()
        ),
        "medical_history": new_medical_history,
        "confirmed_symptoms": confirmed,
        "denied_symptoms": denied,
        "uncertain_symptoms": uncertain,
    }
