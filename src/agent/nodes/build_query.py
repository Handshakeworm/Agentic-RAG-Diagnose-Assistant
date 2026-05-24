"""src/agent/nodes/build_query.py — Agent ② build_query 节点(DEV_SPEC §4.1.2 ②)。

**2026-05-24 删除 Step 1 NER** — 经评估完全冗余:
  - 首轮 chief/present 是 ① info_collect 刚写出来的,同时 ① 已 holistic 抽好 confirmed/denied/uncertain;
    ② 二次 NER 信息源相同(且更窄,看的是 ① 浓缩后),不可能补到 ① 没抓到的症状,
    反而字符串近似匹配(如"右上腹疼痛" vs "右上腹痛")会重复塞污染 confirmed
  - 后续轮 NER 抽了 entities 也只在 `if is_first_round` 才分流 — 死代码,纯 70s 空转
  - check 路径(报告回传后)本来就 skip
  - 删后省 70s/轮,首轮亦省;新增症状全部由 ① / ⓪a + intake_followup_ask / ⑦ 三个采集节点产

剩 2 步:

  Step 1 Sparse 多字段直采 — 不查 terms_collection,纯 state 字段拼接:
                           chief_complaint + slots 单值字段(location / duration_pattern /
                           onset_mode)+ slots list 字段(associated_symptoms / aggravating /
                           relieving / trigger / nature / severity)+ report_findings 的
                           positive_findings(全加)+ impressions(阴性过滤:含 "(-)" / "正常" /
                           "阴性" / "未见" / "无异常" 的整条跳过)
  Step 2 Dense Query 构建 — LLM 整合 confirmed/slots/report_findings → dense_query;
                           sparse_queries 直接照搬 Step 1 产出

LLM 调用 1 处(Step 2 Query),按 §9.1 中安全级模板独立写 try/except/finally + 6 指标。
"""
from __future__ import annotations

import json
import logging
import re
import time

from config.settings import settings
from src.agent.schemas.query_construction import QueryConstructionOutput
from src.agent.state import SLOT_UNKNOWN_SENTINEL, MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import build_query_construction_prompt


# Step 1 阴性 impressions 过滤:含此类字样的 impressions 整条视为阴性,跳过(BM25 不懂否定)
_NEGATIVE_IMPRESSION_RE = re.compile(r"\(-\)|正常|阴性|未见|无异常")

# Step 1 slots 单值字段(每条 strip 后长度 ≥ 2 入 sparse)
_SLOT_SCALAR_FIELDS = (
    "location", "duration_pattern", "onset_mode",
)
# Step 1 slots list 字段(每条独立成袋)
# 2026-05-22:trigger/nature/severity 从 SCALAR 迁过来(schema 改 list[str] 后)
_SLOT_LIST_FIELDS = (
    "associated_symptoms", "aggravating", "relieving",
    "trigger", "nature", "severity",
)


_logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Step 2: Dense Query 构建调用包装(裸 §9.1 模板)
# ────────────────────────────────────────────────────────────────────────────


def _call_query_construction(
    confirmed_symptoms: list[str],
    medical_history_summary: str,
    report_positive: list[str],
    report_impressions: list[str],
    filled_slots: dict,
) -> QueryConstructionOutput:
    # 2026-05-24 sprint:NER 删除后此调用是 Step 2(原 Step 3);并切 flash
    # (dense_query 改写是翻译+整合性任务,不需 reasoner 深度,flash 实测 5~15s vs pro ~85s)
    node, schema = "build_query_step2_query", "QueryConstructionOutput"
    _attempts.labels(node=node, schema=schema).inc()
    t0 = time.perf_counter()
    try:
        chain = (
            get_llm(model=settings.llm.FAST_MODEL_NAME)
            .with_structured_output(QueryConstructionOutput, method="json_mode")
            .with_retry(stop_after_attempt=3)
        )
        return chain.invoke(
            build_query_construction_prompt(
                confirmed_symptoms=confirmed_symptoms,
                medical_history_summary=medical_history_summary,
                report_positive=report_positive,
                report_impressions=report_impressions,
                filled_slots=filled_slots,
            ),
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": node, "schema": schema},
            },
        )
    except Exception as e:
        _failures.labels(
            node=node, schema=schema, exception_type=type(e).__name__
        ).inc()
        _logger.error("[%s] query construction failed: %s", node, e, exc_info=True)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=node, schema=schema).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", node, elapsed)


# ────────────────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────────────────


def _summarize_history(history: dict) -> str:
    """病史 dict → 一行摘要,只取最有诊断意义的项,避免 prompt 膨胀。"""
    parts = []
    past = history.get("past_history") or {}
    if past:
        parts.append(f"既往史:{json.dumps(past, ensure_ascii=False)[:120]}")
    if history.get("medication_history"):
        parts.append(
            f"用药史:{json.dumps(history['medication_history'], ensure_ascii=False)[:80]}"
        )
    if history.get("family_history"):
        parts.append(
            f"家族史:{json.dumps(history['family_history'], ensure_ascii=False)[:80]}"
        )
    return "; ".join(parts)


def build_query(state: MedicalState) -> dict:
    """两步执行:Step 1 sparse 词袋(确定性)+ Step 2 dense_query LLM 改写。

    2026-05-24:删 Step 1 NER(冗余,详见模块 docstring)。confirmed/denied/uncertain 直接
    透传 state(由 ① / ⓪a + intake_followup_ask / ⑦ 三个采集节点维护)。
    """
    confirmed_symptoms = list(state.confirmed_symptoms)
    denied_symptoms = list(state.denied_symptoms)
    uncertain_symptoms = list(state.uncertain_symptoms)

    # ─── Step 1: Sparse 多字段直采(确定性,无 LLM,RETRIEVAL_EVAL §2)───
    # 来源 A:state 多字段(chief_complaint + slots 单值 + slots list)
    # 来源 B:report_findings 的 positive_findings(全加)+ impressions(阴性过滤)
    sparse_queries: list[str] = []

    def _add(item: str | None) -> None:
        if item is None:
            return
        s = item.strip()
        # 哨兵值"(患者未明确)"是 ⑦ 标记"已问无答"的标签,不是症状词,扔给 BM25 只会污染召回
        if s == SLOT_UNKNOWN_SENTINEL:
            return
        if len(s) >= 2:
            sparse_queries.append(s)

    slots_dict = state.present_illness_slots.model_dump()

    # 来源 A.1 — 主诉
    _add(state.chief_complaint)
    # 来源 A.2 — slots 单值字段
    for field in _SLOT_SCALAR_FIELDS:
        _add(slots_dict.get(field))
    # 来源 A.3 — slots list 字段(每条独立成袋)
    for field in _SLOT_LIST_FIELDS:
        for item in slots_dict.get(field) or []:
            _add(item)

    # 来源 B — report_findings;report_pos / report_imp 同时供 Step 2 LLM dense_query 改写
    report_pos: list[str] = []
    report_imp: list[str] = []
    for f in state.report_findings:
        report_pos.extend(f.get("positive_findings") or [])
        report_imp.extend(f.get("impressions") or [])

    for item in report_pos:
        _add(item)
    for item in report_imp:
        if item and _NEGATIVE_IMPRESSION_RE.search(item):
            continue  # 跳过阴性印象(BM25 不懂否定,反向贡献)
        _add(item)

    # 保序去重
    sparse_queries = list(dict.fromkeys(sparse_queries))

    # ─── Step 2: Query 构建(LLM)───

    # 给 dense_query 改写 LLM 的 filled_slots:剥掉哨兵单值,多值列表过滤哨兵元素。
    # 哨兵=已问无答,对检索改写没意义(诊断 prompt 那边会另带哨兵进 LLM 解释语义)
    def _strip_sentinel(value):
        if isinstance(value, list):
            cleaned = [x for x in value if x and x != SLOT_UNKNOWN_SENTINEL]
            return cleaned or None
        if value == SLOT_UNKNOWN_SENTINEL:
            return None
        return value

    filled_slots = {
        k: cleaned
        for k, v in state.present_illness_slots.model_dump().items()
        if v and (cleaned := _strip_sentinel(v))
    }
    history_summary = _summarize_history(state.medical_history)

    qc = _call_query_construction(
        confirmed_symptoms=confirmed_symptoms,
        medical_history_summary=history_summary,
        report_positive=report_pos,
        report_impressions=report_imp,
        filled_slots=filled_slots,
    )

    return {
        "confirmed_symptoms": confirmed_symptoms,
        "denied_symptoms": denied_symptoms,
        "uncertain_symptoms": uncertain_symptoms,
        "dense_query": qc.dense_query,
        # sparse_queries 由 Step 1 确定性产出,LLM 不参与(详见 QueryConstructionOutput docstring)
        "sparse_queries": sparse_queries,
    }
