"""src/agent/nodes/build_query.py — Agent ② build_query 节点(DEV_SPEC §4.1.2 ②)。

三步流程,每轮循环均完整执行:

  Step 1 NER             — LLM 抽取医学实体(首轮:chief + present_illness;后续轮:
                           仅当 followup_round > last_nlu_round 时对 followup_answer
                           NER,跳过空转)
                           首轮把 NER 抽到的 symptom 类(temporality=current)按
                           negation 分流写入 confirmed_symptoms / denied_symptoms,
                           直接用 raw text(无 EL,无 preferred_term 归一化)
  Step 2 Sparse 多字段直采 — 不查 terms_collection,纯 state 字段拼接:
                           chief_complaint + slots 单值字段(trigger / location /
                           nature / severity / duration_pattern / onset_mode)+ slots
                           list 字段(associated_symptoms / aggravating / relieving)
                           + report_findings 的 positive_findings(全加)+ impressions
                           (阴性过滤:含 "(-)" / "正常" / "阴性" / "未见" / "无异常"
                           的整条跳过,避免 BM25 不懂否定造成反向召回)
  Step 3 Dense Query 构建 — LLM 整合 confirmed/slots/report_findings → dense_query;
                           sparse_queries 直接照搬 Step 2 产出

LLM 调用两处(Step 1 NER、Step 3 Query),按 §9.1 中安全级模板独立写
try/except/finally,各自上报 6 指标。
"""
from __future__ import annotations

import json
import logging
import re
import time

from config.settings import settings
from src.agent.schemas.ner import NERResult
from src.agent.schemas.query_construction import QueryConstructionOutput
from src.agent.state import MedicalState
from src.common.metrics import _attempts, _failures, _latency, retry_observer
from src.models.llm_client import get_llm
from src.prompts.agent import (
    build_ner_prompt,
    build_query_construction_prompt,
)


# Step 2 阴性 impressions 过滤:含此类字样的 impressions 整条视为阴性,跳过(BM25 不懂否定)
_NEGATIVE_IMPRESSION_RE = re.compile(r"\(-\)|正常|阴性|未见|无异常")

# Step 2 slots 单值字段(每条 strip 后长度 ≥ 2 入 sparse)
_SLOT_SCALAR_FIELDS = (
    "trigger", "location", "nature", "severity", "duration_pattern", "onset_mode",
)
# Step 2 slots list 字段(每条独立成袋)
_SLOT_LIST_FIELDS = ("associated_symptoms", "aggravating", "relieving")


_logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Step 1: NER 调用包装(裸 §9.1 模板)
# ────────────────────────────────────────────────────────────────────────────


def _call_ner(text: str) -> NERResult:
    node, schema = "build_query_step1_ner", "NERResult"
    _attempts.labels(node=node, schema=schema).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(NERResult, method="json_mode").with_retry(stop_after_attempt=3)
        return chain.invoke(
            build_ner_prompt(text),
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": node, "schema": schema},
            },
        )
    except Exception as e:
        _failures.labels(
            node=node, schema=schema, exception_type=type(e).__name__
        ).inc()
        _logger.error("[%s] NER failed: %s", node, e, exc_info=True)
        raise
    finally:
        _latency.labels(node=node, schema=schema).observe(
            time.perf_counter() - t0
        )


# ────────────────────────────────────────────────────────────────────────────
# Step 3: Query 构建调用包装(裸 §9.1 模板)
# ────────────────────────────────────────────────────────────────────────────


def _call_query_construction(
    confirmed_symptoms: list[str],
    medical_history_summary: str,
    report_positive: list[str],
    report_impressions: list[str],
    filled_slots: dict,
) -> QueryConstructionOutput:
    node, schema = "build_query_step3_query", "QueryConstructionOutput"
    _attempts.labels(node=node, schema=schema).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(QueryConstructionOutput, method="json_mode").with_retry(stop_after_attempt=3)
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
        _latency.labels(node=node, schema=schema).observe(
            time.perf_counter() - t0
        )


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
    """三步执行;若检查路径(followup_round == last_nlu_round)直接跳到 Step 3。"""
    is_first_round = state.followup_round == 0
    is_check_path = (
        not is_first_round and state.followup_round == state.last_nlu_round
    )

    confirmed_symptoms = list(state.confirmed_symptoms)
    denied_symptoms = list(state.denied_symptoms)

    # ─── Step 1: NER(check path 跳过,首轮对 chief+present,后续轮对 answer)───
    if not is_check_path:
        if is_first_round:
            ner_text = (
                f"{state.chief_complaint}\n{state.present_illness}".strip()
            )
        else:
            ner_text = state.followup_answer or ""

        ner_text = ner_text.strip()
        if ner_text:
            ner_result = _call_ner(ner_text)
            entities = ner_result.entities
        else:
            entities = []

        # 首轮主诉症状初始化:symptom 类(temporality=current)按 negation 分流
        # 直接用 NER 原文(无 EL,不做归一化);下游 LLM 能处理口语形式
        if is_first_round:
            for ent in entities:
                if ent.entity_type != "symptom":
                    continue
                if ent.temporality != "current":
                    continue
                text = (ent.text or "").strip()
                if not text:
                    continue
                if ent.negation:
                    if text not in denied_symptoms:
                        denied_symptoms.append(text)
                else:
                    if text not in confirmed_symptoms:
                        confirmed_symptoms.append(text)

    # ─── Step 2: Sparse 多字段直采(确定性,无 LLM,RETRIEVAL_EVAL §2)───
    # 来源 A:state 多字段(chief_complaint + slots 单值 + slots list)
    # 来源 B:report_findings 的 positive_findings(全加)+ impressions(阴性过滤)
    sparse_queries: list[str] = []

    def _add(item: str | None) -> None:
        if item is None:
            return
        s = item.strip()
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

    # 来源 B — report_findings;report_pos / report_imp 同时供 Step 3 LLM dense_query 改写
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

    # ─── Step 3: Query 构建(LLM)───

    filled_slots = {
        k: v for k, v in state.present_illness_slots.model_dump().items() if v
    }
    history_summary = _summarize_history(state.medical_history)

    qc = _call_query_construction(
        confirmed_symptoms=confirmed_symptoms,
        medical_history_summary=history_summary,
        report_positive=report_pos,
        report_impressions=report_imp,
        filled_slots=filled_slots,
    )

    update = {
        "confirmed_symptoms": confirmed_symptoms,
        "denied_symptoms": denied_symptoms,
        "dense_query": qc.dense_query,
        # sparse_queries 由 Step 2 确定性产出,LLM 不参与(详见 QueryConstructionOutput docstring)
        "sparse_queries": sparse_queries,
    }

    # NER 已执行 → 推进游标(spec §4.1.2 ② Step 1)
    if not is_check_path:
        update["last_nlu_round"] = state.followup_round

    return update
