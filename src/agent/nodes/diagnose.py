"""src/agent/nodes/diagnose.py — Agent ⑩ diagnose 节点(DEV_SPEC §4.1.2 ⑩)。

执行顺序:
  Step -1  followup_round 触顶兜底短路(非 LLM,优先级最高)
  Step 0   Cross-Encoder 精排截断 + 写 last_reranked_chunks
  Step 0.5 Context 扩展(spec §3.2.3)— 仅 prompt 用,不写回 State
             规则 1:child → parent_chunk_id 父块全文
             规则 2:table/figure → parent 父块全文 + 自身 image_path 截图
             规则 3:父块 → heading_path_id 同节图表(封顶 RETRIEVE_PARENT_FIGURE_CAP)
  Step 1   1 步 LLM(原生多模态模型,DashScope qwen3.5-plus):直接出 DiagnosisOutput
             figure 的 image_path 转 base64 作为多模态消息送入

设计:1 步 LLM 对齐评测 `.eval/rag_eval/run_diagnose_eval.py`(评测平均 2 分钟、top1
93.5% / top3 100%);原 3 步链(EvidenceSheet → DiagnosisRanking → DiagnosisOutput)整体废弃
— 3 步链让总延迟到 4-6 分钟,且评测证明 1 步信息全给已经能拿到目标精度。

兜底:LLM 调用失败 → 立即停止,产 insufficient + failure_reason="step_1_structured_output_failed",
prompt + raw_output 写入 State 供审计(spec §9.6.2)。
"""
from __future__ import annotations

import logging
import time

from config.settings import settings
from src.agent.schemas.diagnosis import DiagnosisOutput
from src.agent.state import MedicalState
from src.agent.utils.chunks_lookup import (
    lookup_chunk_content,
    lookup_figures_by_heading_path,
)
from src.agent.utils.report_loader import load_report
from src.common.metrics import (
    _attempts,
    _diagnose_reason,
    _failures,
    _fallbacks,
    _latency,
    retry_observer,
)
from src.models.llm_client import get_llm
from src.prompts.agent import build_diagnose_prompt
from src.rag.retrieval.reranker import rerank_with_fallback


_logger = logging.getLogger(__name__)

_NODE = "diagnose"
_SCHEMA = "DiagnosisOutput"


# ────────────────────────────────────────────────────────────────────────────
# 兜底产出
# ────────────────────────────────────────────────────────────────────────────


def _capped_result() -> list[dict]:
    return [
        {
            "disease": "信息不足以支持可靠诊断",
            "probability": 0.0,
            "evidence": ["追问轮次达上限 MAX_FOLLOWUP_ROUNDS"],
            "differentiation": None,
            "differentiation_type": "insufficient",
            "failure_reason": "followup_round_capped",
        }
    ]


def _llm_failure_result(exc: BaseException) -> list[dict]:
    return [
        {
            "disease": "信息不足以支持可靠诊断",
            "probability": 0.0,
            "evidence": ["Step 1 结构化输出失败"],
            "differentiation": None,
            "differentiation_type": "insufficient",
            "failure_reason": (
                f"step_1_structured_output_failed: {type(exc).__name__}: {exc}"
            ),
        }
    ]


# ────────────────────────────────────────────────────────────────────────────
# Step 0 / 0.5 工具
# ────────────────────────────────────────────────────────────────────────────


def _candidate_text(chunk: dict) -> str:
    parts = []
    for vh in chunk.get("vector_hits") or []:
        mt = (vh.get("matched_text") or "").strip()
        if mt:
            parts.append(mt)
    return " ".join(parts)


def _rerank_and_truncate(
    candidate_chunks: list[dict], query: str, top_k: int
) -> tuple[list[dict], list[str]]:
    """Step 0:reranker.rerank_with_fallback → 重排截断。返回 (reranked_chunks, indexed_text)。

    fallback 路径(reranker 不可用 / 超时)→ 取原序前 top_k(spec §3.2.3 强约束:
    精排不抛异常,失败必走 fallback)。
    """
    if not candidate_chunks:
        return [], []
    documents = [_candidate_text(c) or c.get("source_chunk_id", "") for c in candidate_chunks]
    indices = rerank_with_fallback(
        query=query,
        documents=documents,
        top_k=top_k,
        timeout_sec=settings.reranker.TIMEOUT_SECONDS,
        enabled=settings.reranker.ENABLED,
    )
    reranked = [candidate_chunks[i] for i in indices]
    text = [documents[i] for i in indices]
    return reranked, text


def _load_figure_data_uri(image_path: str | None) -> str | None:
    """把 chunks.image_path 加载成 base64 data URI(失败返 None 不抛)。"""
    if not image_path:
        return None
    try:
        loaded = load_report(image_path)
    except Exception as e:
        _logger.warning("figure image load failed (%s): %s", image_path, e)
        return None
    if loaded.get("kind") != "image":
        return None  # PDF 等非图(理论上不该是 figure image_path,防御性兜底)
    return loaded.get("data_uri")


def _build_diagnose_context(
    reranked_chunks: list[dict], reranked_text: list[str]
) -> dict:
    """Step 0.5:spec §3.2.3 Context 扩展,产出供 prompt 消费的结构(父块全文 + 同节图表)。

    规则 1:child → parent_chunk_id 父块全文(替换 reranked_text 中的小块)
    规则 2:table/figure 自身 → 父块全文(同规则 1)+ figure 自身截图加载
    规则 3:父块 → heading_path_id 同节图表(封顶 RETRIEVE_PARENT_FIGURE_CAP)

    去重:三条规则展开后按 chunk_id 去重(常见 case 是图表 chunk 直接命中 +
    父块展开后规则 3 又拉同一图表,只留一份)。

    medical_statement 不进 prompt(spec §3.1.5.1 + §3.2.3:enrichment 字段仅承担召回
    辅助,不作 LLM context payload)。

    Returns:
        {
            "parent_texts": list[str],   # 与 reranked_chunks 同序
            "figures": list[dict],       # 跨规则去重后的同节 + 直接命中图表
        }
    """
    if not reranked_chunks:
        return {"parent_texts": list(reranked_text), "figures": []}

    chunk_ids = [c.get("source_chunk_id") for c in reranked_chunks if c.get("source_chunk_id")]
    try:
        meta = lookup_chunk_content(chunk_ids)
    except Exception as e:
        _logger.warning("chunks_lookup failed during context build: %s", e)
        return {"parent_texts": list(reranked_text), "figures": []}

    # ── 规则 1 + 2:展开父块文本(与 reranked_chunks 同序)──
    parent_text_by_idx: list[str | None] = []
    parent_heading_paths: set[str] = set()
    direct_hit_figures: list[dict] = []
    parent_ids_to_fetch: list[str] = []

    for chunk, fallback_text in zip(reranked_chunks, reranked_text):
        cid = chunk.get("source_chunk_id")
        info = meta.get(cid) if cid else None
        if not info:
            parent_text_by_idx.append(fallback_text)
            continue

        # 规则 2 一部分:直接命中 table/figure → 记到 direct_hit_figures
        if info.get("chunk_type") in ("table", "figure"):
            direct_hit_figures.append(
                {
                    "chunk_id": cid,
                    "chunk_type": info["chunk_type"],
                    "chunk_raw_text": info.get("chunk_raw_text") or "",
                    "title": info.get("title"),
                    "image_data_uri": _load_figure_data_uri(info.get("image_path")),
                }
            )

        # 规则 1 + 规则 2 父块替换:有 parent_chunk_id 则取父块全文
        parent_id = info.get("parent_chunk_id")
        if parent_id:
            parent_ids_to_fetch.append(parent_id)
            parent_text_by_idx.append(None)  # 占位,下一轮回填
        else:
            # 父块缺失或自己就是父块 → 用 chunk_raw_text 兜底
            body = info.get("chunk_raw_text") or fallback_text
            parent_text_by_idx.append(body)
            if info.get("heading_path_id"):
                parent_heading_paths.add(info["heading_path_id"])

    # 一次性批量查父块
    parent_meta: dict[str, dict] = {}
    if parent_ids_to_fetch:
        unique_parent_ids = list({pid for pid in parent_ids_to_fetch if pid not in meta})
        try:
            parent_meta = lookup_chunk_content(unique_parent_ids) if unique_parent_ids else {}
        except Exception as e:
            _logger.warning("parent chunks lookup failed: %s", e)
            parent_meta = {}
        parent_meta = {**parent_meta, **{k: v for k, v in meta.items() if k in parent_ids_to_fetch}}

    # 回填父块文本占位
    for i, chunk in enumerate(reranked_chunks):
        if parent_text_by_idx[i] is not None:
            continue
        cid = chunk.get("source_chunk_id")
        info = meta.get(cid) if cid else None
        parent_id = info.get("parent_chunk_id") if info else None
        parent_info = parent_meta.get(parent_id) if parent_id else None
        if parent_info and parent_info.get("chunk_raw_text"):
            parent_text_by_idx[i] = parent_info["chunk_raw_text"]
            if parent_info.get("heading_path_id"):
                parent_heading_paths.add(parent_info["heading_path_id"])
        else:
            body = (info.get("chunk_raw_text") if info else None) or reranked_text[i]
            parent_text_by_idx[i] = body or ""

    # ── 规则 3:按 heading_path_id 批量查同节图表(封顶 RETRIEVE_PARENT_FIGURE_CAP)──
    cap = settings.agent_limits.RETRIEVE_PARENT_FIGURE_CAP
    same_section_figures: list[dict] = []
    if parent_heading_paths:
        try:
            grouped = lookup_figures_by_heading_path(parent_heading_paths, cap=cap)
        except Exception as e:
            _logger.warning("same-section figures lookup failed: %s", e)
            grouped = {}
        for figs in grouped.values():
            for f in figs:
                same_section_figures.append(
                    {
                        "chunk_id": f["chunk_id"],
                        "chunk_type": f["chunk_type"],
                        "chunk_raw_text": f.get("chunk_raw_text") or "",
                        "title": f.get("title"),
                        "image_data_uri": _load_figure_data_uri(f.get("image_path")),
                    }
                )

    # ── 去重:跨规则 2/3 按 chunk_id 去重 ──
    seen: set[str] = set()
    merged_figures: list[dict] = []
    for f in (*direct_hit_figures, *same_section_figures):
        cid = f.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        merged_figures.append(f)

    return {
        "parent_texts": [t or "" for t in parent_text_by_idx],
        "figures": merged_figures,
    }


# ────────────────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────────────────


def diagnose(state: MedicalState) -> dict:
    # ─── Step -1: 追问触顶兜底 ───
    if state.followup_round >= settings.agent_limits.MAX_FOLLOWUP_ROUNDS:
        _diagnose_reason.labels(reason_kind="followup_round_capped").inc()
        return {
            "diagnosis_result": _capped_result(),
            # 兜底路径不动 last_reranked_chunks / last_diagnose_prompt / raw_output
            # (前者 init 是空 list,后两者 init 是 None,符合"正常路径保持初始值")
        }

    # ─── Step 0: 精排截断 ───
    rerank_top_k = settings.retrieval.RERANK_TOP_K
    rerank_query = " / ".join(
        [state.chief_complaint or ""] + state.confirmed_symptoms
    ).strip()
    reranked_chunks, reranked_text = _rerank_and_truncate(
        state.candidate_chunks, rerank_query, rerank_top_k
    )

    # ─── Step 0.5: Context 扩展(spec §3.2.3,仅 prompt 用,不写回 State)───
    ctx = _build_diagnose_context(reranked_chunks, reranked_text)

    # ─── Step 1: 1 步 LLM(原生多模态,DashScope qwen3.5-plus)───
    # medical_history 不再 json 截断:build_diagnose_prompt 内部用 _format_medical_history
    # 结构化展开 8 个子项,空字段显式标"未询问 ≠ 阴性"(对齐评测口径)
    slots_dict = state.present_illness_slots.model_dump()

    vision_llm = get_llm(
        model=settings.llm.VISION_MODEL_NAME,
        base_url=settings.llm.VISION_BASE_URL,
        api_key=settings.llm.VISION_API_KEY,
    )
    chain = vision_llm.with_structured_output(
        DiagnosisOutput, method="json_mode"
    ).with_retry(stop_after_attempt=3)

    messages, prompt_text = build_diagnose_prompt(
        parent_texts=ctx["parent_texts"],
        figures=ctx["figures"],
        chief_complaint=state.chief_complaint,
        present_illness=state.present_illness,
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
        uncertain_symptoms=state.uncertain_symptoms,
        slots=slots_dict,
        medical_history=state.medical_history,
        report_findings=state.report_findings,
        unaskable_symptoms=state.unaskable_symptoms,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    last_raw_output: str | None = None
    try:
        result: DiagnosisOutput = chain.invoke(
            messages,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA},
            },
        )
        last_raw_output = result.model_dump_json()
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _fallbacks.labels(node=_NODE, fallback_type="insufficient").inc()
        _diagnose_reason.labels(reason_kind="step_1_failed").inc()
        _logger.error(
            "diagnose pipeline failed: %s: %s", type(e).__name__, e, exc_info=True,
        )
        return {
            "diagnosis_result": _llm_failure_result(e),
            "last_reranked_chunks": reranked_chunks,
            "last_diagnose_prompt": prompt_text,
            "last_diagnose_raw_output": str(e),
        }
    finally:
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(
            time.perf_counter() - t0
        )

    # 正常路径产出 — retained_unaskable 覆盖 ④ 写的粗筛版,供 ⑧a recommend_exam 消费
    return {
        "diagnosis_result": [r.model_dump() for r in result.results],
        "unaskable_symptoms": [u.model_dump() for u in result.retained_unaskable],
        "last_reranked_chunks": reranked_chunks,
        # 正常路径保持 None(spec §9.6.2)— 不写 last_diagnose_prompt / raw_output
    }
