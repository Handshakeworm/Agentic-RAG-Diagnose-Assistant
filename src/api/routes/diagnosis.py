"""问诊接口(DEV_SPEC §8.4 G4 + §9.6)。

`POST /diagnose` 走 **SSE 流式**:每跑完一个节点推一条 `progress` event(供前端
"灰字告诉用户系统进行到哪一步了"),最后推 `interrupt`(追问/检查回传)
或 `completed`(终态)或 `error`(异常)。

为什么不是 JSON 一发一收:用户等待 5~15s 期间需要"有事干 + 安全感"的进度反馈,
LangGraph 的 `astream(stream_mode="updates")` 天然按 super-step 逐节点 yield,
直接转成 SSE 即可。

事件协议(`data: <json>\\n\\n`):
- `{event: "progress", node, text}`               每个非 interrupt 节点完成后
- `{event: "interrupt", session_id, status, ...}` 走到 wait_followup_answer / wait_exam_report
- `{event: "completed", session_id, status, ...}` 终态(format_response 出 END)
- `{event: "error",     session_id, detail}`      任意节点抛异常

终态写 rag_trace + conversations 与原同步实现完全一致(§9.6.2 / §9.6.5 裸代码
模板),只是发生时机改到 generator 内最后一个 yield 之前。

⚠️ TODO(留给用户拍板):
- checkpointer 当前用 `InMemorySaver` 模块级单例 — 进程重启会丢 in-flight session;
  生产应换 `langgraph-checkpoint-postgres.PostgresSaver`(deps 已有),
  需要单独建 `langgraph_checkpoints` 表 + Alembic 迁移
- conversations 表本端只在 completed 终态写一行 user_input=raw / llm_output=final;
  spec §2.4.3 一行 = 一次"用户-系统交互",理论上每个追问轮也算一次,
  但首版收口在终态,降低中途异常时的脏数据
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from sqlalchemy.orm import Session as OrmSession

from config.settings import settings
from src.agent.graph import build_app
from src.agent.state import MedicalState
from src.api.middleware.auth_middleware import CurrentUser, get_current_user
from src.api.routes.auth import get_db  # 复用 G2 的 session Depends
from src.api.schemas.diagnosis_schema import DiagnoseRequest
from src.db.postgres.models_audit import RagTrace
from src.db.postgres.models_dialog import Conversation, Session as SessionRow


_logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────────────────────────────────────
# 节点 → 用户可见进度文案(灰字)
# ────────────────────────────────────────────────────────────────────────────
# 文案放后端一处统一维护,前端只渲染 text;不在此处的节点(wait_*)走 interrupt
# event,不推 progress(避免"等待您的回答"被误当成系统在干活)。
_NODE_PROGRESS_TEXT: dict[str, str] = {
    "initial_ask":                   "正在加载您的档案…",
    "info_collect":                  "正在理解您的主诉…",
    "analyze_initial_reports":       "正在分析检查报告…",
    "intake_followup_ask":           "正在准备需要确认的细节…",
    "generate_followup":             "正在准备要确认的内容…",  # Step A pro 决策;Step B 由 custom event 切灰字
    "process_followup_answer":       "正在解析您的回答…",
    "build_query":                   "正在构造检索查询…",
    "retrieve":                      "正在检索医学知识库…",
    "select_discriminative_symptom": "正在分析关键症状…",
    "diagnose":                      "正在综合诊断(三步推理)…",
    "recommend_exam":                "正在评估需要的检查…",
    "process_exam_result":           "正在处理检查结果…",
    "safety_gate":                   "正在做安全核查…",
    "generate_advice":               "正在生成用药建议…",
    "format_response":               "正在整理回复…",
}

# 节点内 sub-step 灰字:节点用 langchain `dispatch_custom_event("progress_step", {"step": ...})`
# 推,SSE 在 astream_events `on_custom_event` 分支接,name 必须是 "progress_step"。
# 用途:同一节点内多步 LLM 时,把每步 LLM 开始前的灰字单独显示,而不是节点 chain_start
# 只 fire 一次后整个节点过程都一个灰字。
_CUSTOM_STEP_PROGRESS_TEXT: dict[str, str] = {
    "intake_translate_answers": "正在处理回答…",  # intake 4 batch 收完后调 parse_followup_response 前
    "step_b_question_gen":      "正在生成追问…",  # ⑤ Step A 决策完,Step B flash 拼问句前
}


# ────────────────────────────────────────────────────────────────────────────
# Compiled graph 单例 — checkpointer 跨请求保存 interrupt 状态
# ────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_compiled_graph():
    """模块级 compiled graph + InMemorySaver。

    interrupt → resume 必须用同一个 checkpointer 实例,所以走单例。
    """
    return build_app()


# ────────────────────────────────────────────────────────────────────────────
# 辅助:rag_trace error_info 派生(spec §9.6.3) + SSE 编码
# ────────────────────────────────────────────────────────────────────────────


def _build_error_info(diagnosis_result: list[dict]) -> dict | None:
    """spec §9.6.3 的精确转写。正常 LLM 推理 → None;系统级失败 → 结构化 dict。"""
    if not diagnosis_result:
        return {"source": "diagnose", "failure_reason": "empty_diagnosis_result", "step": None}
    reason = diagnosis_result[0].get("failure_reason")
    if reason is None:
        return None
    step = None
    if reason.startswith("step_") and "_structured_output_failed" in reason:
        try:
            step = int(reason.split("_")[1])
        except (IndexError, ValueError):
            step = None
    return {"source": "diagnose", "failure_reason": reason, "step": step}


def _sse(payload: dict) -> str:
    """SSE 一条消息:`data: <json>\\n\\n`(空行表示消息结束,符合 EventSource 协议)。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _to_dict(v) -> dict | None:
    """state_dict 字段在 LangGraph 反序列化后可能是 Pydantic 对象(SessionLatencyMs /
    SessionTokenUsage 等)或 dict。统一转 dict 供 ** 解构 / JSON 序列化使用。"""
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if hasattr(v, "model_dump"):
        return v.model_dump()
    return dict(v)


# ────────────────────────────────────────────────────────────────────────────
# 主端点
# ────────────────────────────────────────────────────────────────────────────


@router.post(
    "/diagnose",
    summary="问诊主接口(SSE 流式,支持多轮追问 / 检查回传)",
)
async def diagnose(
    req: DiagnoseRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: OrmSession = Depends(get_db),
) -> StreamingResponse:
    # ─── 1. session 寻址或新建(同步操作,放 generator 外提早暴露 4xx) ───
    if req.session_id:
        sess: SessionRow | None = db.get(SessionRow, req.session_id)
        if sess is None:
            raise HTTPException(404, f"session {req.session_id} 不存在")
        if sess.user_id != current_user.user_id:
            # 不能蹭别人的 session_id 接着跑
            raise HTTPException(403, "session 不属于当前用户")
        session_id = sess.id
        is_first_round = False
    else:
        if not req.patient_input:
            raise HTTPException(422, "首次问诊必须提供 patient_input")
        sess = SessionRow(user_id=current_user.user_id, title=req.patient_input[:60])
        db.add(sess)
        db.flush()
        db.refresh(sess)
        session_id = sess.id
        is_first_round = True

    # ─── 2. 构造 graph invoke 入参 ────────────────────────────────────
    graph_app = _get_compiled_graph()
    config = {"configurable": {"thread_id": f"session_{session_id}"}}

    if is_first_round:
        graph_input: MedicalState | Command = MedicalState(
            patient_id=current_user.user_id,
            patient_input=req.patient_input or "",
        )
    else:
        if req.followup_answer is not None:
            graph_input = Command(resume=req.followup_answer)
        elif req.exam_results is not None:
            graph_input = Command(resume=req.exam_results)
        else:
            raise HTTPException(
                422,
                "后续轮必须提供 followup_answer 或 exam_results 之一",
            )

    # ─── 3. SSE generator:逐节点推 progress,终态/interrupt/异常推终止 event ───
    async def event_stream() -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        try:
            # 改用 astream_events(version="v2"):比 stream_mode="updates" 多了 on_chain_start
            # 事件 —— 节点**进入时**触发,灰字推送和实际节点执行同步,不再延迟一拍卡在上一个文案。
            # 过滤条件:event=="on_chain_start" 且 name 在 mapping 里(LangGraph 把每个 node
            # 注册为 chain,name = add_node() 时给的名字;sub-runnable 不在 mapping 不会推)。
            async for ev in graph_app.astream_events(
                graph_input, config=config, version="v2"
            ):
                ev_type = ev.get("event")
                if ev_type == "on_chain_start":
                    name = ev.get("name") or ""
                    text = _NODE_PROGRESS_TEXT.get(name)
                    if text:
                        yield _sse({"event": "progress", "node": name, "text": text})
                elif ev_type == "on_custom_event" and ev.get("name") == "progress_step":
                    # 节点内 sub-step 灰字(intake 末尾翻译 / ⑤ Step B 等)
                    step = (ev.get("data") or {}).get("step", "")
                    text = _CUSTOM_STEP_PROGRESS_TEXT.get(step)
                    if text:
                        yield _sse({"event": "progress", "node": step, "text": text})

            invoke_latency_ms = int((time.perf_counter() - t0) * 1000)

            # ─── 4. astream_events 退出 → 要么 interrupt 要么 END ───
            # **LangGraph 1.1.10 multi-interrupt 行为**:同一个节点内第 2+ 次 interrupt
            # 暂停时,`snapshot.next` 是空 tuple,但 `snapshot.tasks[*].interrupts` 里
            # 仍有 pending interrupt。只看 snapshot.next 会误判 graph 已结束。所以这里
            # 优先从 pending interrupt task 取 next_node + has_pending。
            snapshot = await graph_app.aget_state(config)
            pending_task = next(
                (t for t in (snapshot.tasks or []) if getattr(t, "interrupts", None)),
                None,
            )
            if pending_task is not None:
                has_pending = True
                next_node = pending_task.name
            elif snapshot.next:
                has_pending = True
                next_node = snapshot.next[0]
            else:
                has_pending = False
                next_node = ""

            state_dict = (
                snapshot.values
                if isinstance(snapshot.values, dict)
                else snapshot.values.model_dump()
            )

            if has_pending:
                if next_node in ("initial_ask", "intake_followup_ask"):
                    # ⓪a 节点入口 interrupt(综合 form);intake 节点内 multi-interrupt(slot batch form)
                    # 两者都把 payload 塞在 interrupt(...) 调用里(state 还没 commit)→ 从
                    # snapshot.tasks[*].interrupts[*].value 拿;status 区分前端文案
                    payload: dict = {}
                    for task in (snapshot.tasks or []):
                        for it in (getattr(task, "interrupts", None) or []):
                            if isinstance(getattr(it, "value", None), dict):
                                payload = it.value
                                break
                        if payload:
                            break
                    status = (
                        "ongoing_initial_ask" if next_node == "initial_ask"
                        else "ongoing_followup"  # intake batch 跟普通追问共用前端渲染
                    )
                    yield _sse({
                        "event": "interrupt",
                        "session_id": session_id,
                        "status": status,
                        "pending_question": payload.get("followup_question") or "",
                        "pending_questions": list(payload.get("followup_questions") or []),
                    })
                elif next_node == "wait_followup_answer":
                    yield _sse({
                        "event": "interrupt",
                        "session_id": session_id,
                        "status": "ongoing_followup",
                        "pending_question": state_dict.get("followup_question") or "",
                        "pending_questions": list(state_dict.get("followup_questions") or []),
                    })
                elif next_node == "wait_exam_report":
                    # 2026-05-22:推 recommended_test_groups(分组 + UI 分框);
                    # recommended_tests 留空(由 ⑫ generate_advice 写,⑬ format_response 用)
                    yield _sse({
                        "event": "interrupt",
                        "session_id": session_id,
                        "status": "ongoing_exam",
                        "recommended_test_groups": list(state_dict.get("recommended_test_groups") or []),
                    })
                elif next_node == "analyze_initial_reports":
                    # ①.5 节点内 interrupt:问"要不要上传报告"。payload 文案见
                    # analyze_initial_reports.INITIAL_REPORT_UPLOAD_PROMPT(单点维护)
                    from src.agent.nodes.analyze_initial_reports import (
                        INITIAL_REPORT_UPLOAD_PROMPT,
                    )
                    yield _sse({
                        "event": "interrupt",
                        "session_id": session_id,
                        "status": "ongoing_report_upload",
                        "pending_questions": [INITIAL_REPORT_UPLOAD_PROMPT],
                    })
                else:
                    # 不预期的 interrupt 节点 — 防御日志,按追问处理(总比 500 强)
                    _logger.warning(
                        "session %s interrupt at unexpected node %s",
                        session_id, next_node,
                    )
                    yield _sse({
                        "event": "interrupt",
                        "session_id": session_id,
                        "status": "ongoing_followup",
                        "pending_question": state_dict.get("followup_question") or "(等待用户输入)",
                    })
                return

            # ─── 5. 终态:按 §9.6.5 裸代码模板写 rag_trace + conversation ─────
            s = state_dict
            trace_row = RagTrace(
                session_id=session_id,
                user_id=current_user.user_id,
                raw_query=s.get("patient_input") or "",
                intent_result={
                    "chief_complaint": s.get("chief_complaint"),
                    "confirmed_symptoms": s.get("confirmed_symptoms") or [],
                    "denied_symptoms": s.get("denied_symptoms") or [],
                    "standardized_entities": s.get("standardized_entities") or [],
                },
                retrieved_chunks=s.get("candidate_chunks") or [],
                reranked_chunks=s.get("last_reranked_chunks") or [],
                final_prompt=s.get("last_diagnose_prompt"),
                llm_raw_output=s.get("last_diagnose_raw_output"),
                final_response=s.get("final_response") or "",
                model_name=settings.llm.MODEL_NAME,
                token_usage=_to_dict(s.get("session_token_usage")) or {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
                },
                latency_ms={
                    **(_to_dict(s.get("session_latency_ms")) or {}),
                    "total": invoke_latency_ms,
                },
                error_info=_build_error_info(s.get("diagnosis_result") or []),
            )
            conv_row = Conversation(
                session_id=session_id,
                user_id=current_user.user_id,
                user_input=s.get("patient_input") or "",
                llm_output=s.get("final_response") or "",
                rag_context={
                    "chunk_ids": [
                        c.get("source_chunk_id")
                        for c in (s.get("last_reranked_chunks") or [])
                        if c.get("source_chunk_id")
                    ],
                },
            )
            # spec §9.6.1 事务性:写失败不阻塞响应,但 error 日志告警
            try:
                db.add(trace_row)
                db.add(conv_row)
                db.flush()
            except Exception:
                db.rollback()
                _logger.error(
                    "rag_trace / conversation write failed for session %s",
                    session_id,
                    exc_info=True,
                )

            yield _sse({
                "event": "completed",
                "session_id": session_id,
                "status": "completed",
                "final_response": s.get("final_response"),
                "diagnosis_result": s.get("diagnosis_result") or [],
                "medication_advice": s.get("medication_advice") or [],
                "risk_warnings": s.get("risk_warnings") or [],
            })

        except Exception as e:
            _logger.error(
                "graph stream failed for session %s: %s", session_id, e, exc_info=True
            )
            # SSE 已经发了头,不能 raise HTTPException;吐一条 error event 让前端兜底
            yield _sse({
                "event": "error",
                "session_id": session_id,
                "detail": "诊断服务暂不可用,请稍后再试",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
