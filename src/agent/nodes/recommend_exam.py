"""src/agent/nodes/recommend_exam.py — Agent ⑧a recommend_exam(DEV_SPEC §4.1.2 ⑧)。

**双模式**(看 `state.diagnosis_result` 是否非空切):
- **首诊模式**(⑤ 触发,`diagnosis_result` 为空):⑤ 已经定了"该查什么"(写在
  `unaskable_symptoms`),⑧a 透传消费 → 医生侧 description 转译成患者友好的检查清单
- **鉴别模式**(⑩ 后 `need_exam`,`diagnosis_result` 非空):基于候选 + `retained_unaskable`
  推针对性补漏

**不再读 `candidate_chunks`** — 医学推理已在 ⑤/⑩ 完成,⑧a 只做"医生侧 description →
患者友好文案 + 优先级排序",prompt 短延迟低。

设计目的:首诊模式让 patient 第一轮就拿到该做的全套检查,做完回传后 ⑩ 大概率
一次诊断结案;鉴别模式仅补漏,降低 ⑩ × 2 的概率。

中安全等级失败处理:抛异常终止会话(检查推荐失败说明 LLM 完全不可用)。
"""
from __future__ import annotations

import logging
import time

from src.agent.schemas.recommend_exam import RecommendExamOutput
from src.agent.state import MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_recommend_exam_prompt


_logger = logging.getLogger(__name__)
_NODE = "recommend_exam"
_SCHEMA = "RecommendExamOutput"


def recommend_exam(state: MedicalState) -> dict:
    mode = "differential" if state.diagnosis_result else "intake"
    prompt = build_recommend_exam_prompt(
        mode=mode,
        chief_complaint=state.chief_complaint,
        present_illness=state.present_illness,
        diagnosis_results=list(state.diagnosis_result),
        unaskable_symptoms=list(state.unaskable_symptoms),
        existing_report_findings=list(state.report_findings),
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(RecommendExamOutput, method="json_mode").with_retry(stop_after_attempt=3)
        result: RecommendExamOutput = chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA, "mode": mode},
            },
        )
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _logger.error("[%s] structured output failed (mode=%s): %s", _NODE, mode, e)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", _NODE, elapsed)

    # 去重保留顺序(LLM 偶尔会重复推荐;state 字段定义无重复语义)
    tests_unique: list[str] = []
    seen: set[str] = set()
    for t in result.tests:
        t_clean = (t or "").strip()
        if t_clean and t_clean not in seen:
            tests_unique.append(t_clean)
            seen.add(t_clean)

    # DEBUG remove: 看 ⑧a LLM 真实出口,定位 UI"建议检查清单为空" bug
    _logger.info(
        "[debug] ⑧a output: mode=%s raw_tests=%s tests_unique=%s rationale=%r",
        mode, list(result.tests), tests_unique, result.rationale,
    )

    return {
        "recommended_tests": tests_unique,
        "exam_round": state.exam_round + 1,
    }
