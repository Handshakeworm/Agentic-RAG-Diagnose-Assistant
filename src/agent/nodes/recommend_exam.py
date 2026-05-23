"""src/agent/nodes/recommend_exam.py — Agent ⑧a recommend_exam(DEV_SPEC §4.1.2 ⑧)。

**双模式**(看 `state.diagnosis_result` 是否非空切):
- **首诊模式**(⑤ 触发,`diagnosis_result` 为空):⑤ 已经定了"该查什么"(写在
  `unaskable_symptoms`),⑧a 透传消费 → 医生侧 description 转译成患者友好的检查清单
- **鉴别模式**(⑩ 后 `need_exam`,`diagnosis_result` 非空):基于候选 + `retained_unaskable`
  推针对性补漏

**不再读 `candidate_chunks`** — 医学推理已在 ⑤/⑩ 完成,⑧a 只做"医生侧 description →
患者友好文案 + 按报告载体分组",prompt 短延迟低。

2026-05-22 改造:
- 输出从 `tests: list[str]` → `test_groups: list[ExamGroup]`,按报告载体分组
  (抽血合 1 组 / 体格合 1 组 / 影像各自独立 / 心电图等独立),UI 据此给每组出
  独立上传框,⑨ 解析时拿 group_label 作 hint
- 模型从 pro → flash:任务本质是"翻译 + 整理 + 规则性分组",不需要 reasoner 深度;
  flash 5-15s 完成(对比 pro ~55s),嵌套 schema 输出稳定(同 ⑤ Step A 经验)
- state 写 `recommended_test_groups`(新字段),不再写 `recommended_tests`
  (后者保留给 ⑫ generate_advice 写扁平自然语言建议,⑬ format_response 用)

设计目的:首诊模式让 patient 第一轮就拿到该做的全套检查,做完回传后 ⑩ 大概率
一次诊断结案;鉴别模式仅补漏,降低 ⑩ × 2 的概率。

中安全等级失败处理:抛异常终止会话(检查推荐失败说明 LLM 完全不可用)。
"""
from __future__ import annotations

import logging
import time

from config.settings import settings
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
        # 2026-05-22:pro → flash(任务本质翻译+整理+规则性分组,不需 reasoner 深度)
        chain = get_llm(model=settings.llm.FAST_MODEL_NAME).with_structured_output(
            RecommendExamOutput, method="json_mode"
        ).with_retry(stop_after_attempt=3)
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
        _logger.info("[%s] flash elapsed=%.2fs", _NODE, elapsed)

    # 去重保留顺序 + 兜底过滤(LLM 偶尔产空 group 或重名)
    groups_clean: list[dict] = []
    seen_labels: set[str] = set()
    for g in result.test_groups:
        label = (g.group_label or "").strip()
        items = [i.strip() for i in (g.items or []) if i and i.strip()]
        if not label or not items:
            continue  # 跳过空 group
        if label in seen_labels:
            continue  # 跳过重名 group(LLM 偶发)
        groups_clean.append({
            "group_label": label,
            "items": items,
            "note": (g.note or "").strip(),
        })
        seen_labels.add(label)

    # DEBUG remove: 看 ⑧a 分组输出,验证 flash 是否按规则分对
    _logger.info(
        "[debug] ⑧a output: mode=%s groups=%s rationale=%r",
        mode, groups_clean, result.rationale,
    )

    return {
        "recommended_test_groups": groups_clean,
        "exam_round": state.exam_round + 1,
    }
