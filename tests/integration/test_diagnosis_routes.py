"""tests/integration/test_diagnosis_routes.py — G4 POST /diagnose 闭环。

接口走 **SSE 流式**(`text/event-stream`),test 用 `_read_sse_events` 把响应正文按
`data: <json>\\n\\n` 切回 dict 列表,断言最后一条终止 event(`completed` / `interrupt` /
`error`),其余 `progress` event 不参与功能断言。

graph 用 mock 替换(真跑会调 LLM/Embedding/Reranker,慢 + 烧 token)。mock 同时
桩 `astream`(async generator yield 节点 update)和 `aget_state`(snapshot.next 决定
终止 event 类型)。

需 PG 真服务在跑 + alembic upgrade head。
"""
from __future__ import annotations

import json
import os
import socket
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text


def _pg_alive() -> bool:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _pg_alive(), reason="PG 不可达")


# ────────────────────────────────────────────────────────────────────────────
# fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    from src.api.app import app
    return TestClient(app)


@pytest.fixture
def patient_token():
    """注册一个 patient 用户,返 (token, user_id, email);teardown 级联清。"""
    from src.db.postgres.connection import session_scope

    email = f"g4_{uuid.uuid4().hex[:8]}@example.com"
    from src.api.app import app as fastapi_app
    with TestClient(fastapi_app) as c:
        resp = c.post(
            "/auth/register",
            json={"email": email, "password": "hunter22", "role": "patient"},
        )
    assert resp.status_code == 201
    token = resp.json()["access_token"]

    with session_scope() as s:
        user_id = s.execute(
            text("SELECT id FROM users WHERE email = :e"), {"e": email}
        ).scalar_one()

    yield token, str(user_id), email

    # 级联清:diagnosis_feedback → rag_trace → conversations → sessions → users
    with session_scope() as s:
        s.execute(text("DELETE FROM rag_trace WHERE user_id = :uid"), {"uid": user_id})
        s.execute(text("DELETE FROM conversations WHERE user_id = :uid"), {"uid": user_id})
        s.execute(text("DELETE FROM sessions WHERE user_id = :uid"), {"uid": user_id})
        s.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})


def _make_astream_events(events: list[dict]):
    """构造 graph.astream_events 替身:被调用即返一个 async iterator,逐条 yield event。

    `astream_events` 的真签名:`def astream_events(input, config=..., version=...) -> AsyncIterator[dict]`。
    最小可用 event 形态 `{"event": "on_chain_start", "name": "<node>"}` —— diagnosis.py 只用这两个字段。
    """
    async def _astream_events(*args, **kwargs):
        for e in events:
            yield e
    return _astream_events


def _node_start_events(node_names: list[str]) -> list[dict]:
    """方便构造一连串"节点 X 进入"事件,模拟 graph 走过几个节点。"""
    return [{"event": "on_chain_start", "name": n} for n in node_names]


def _mock_graph_completed(
    final_state: dict, events: list[dict] | None = None
) -> MagicMock:
    """mock graph 立刻终态:astream_events 跑完 → aget_state.next 为空 → 路由进 completed 分支。"""
    g = MagicMock()
    # 默认推一条 format_response 进入事件,SSE 至少会出一条 progress(便于断言节点确实被流出)
    g.astream_events = _make_astream_events(
        events or _node_start_events(["format_response"])
    )
    snapshot = MagicMock(values=final_state, next=())
    g.aget_state = AsyncMock(return_value=snapshot)
    return g


def _mock_graph_interrupt(
    state_dict: dict,
    next_node: str,
    events: list[dict] | None = None,
    interrupt_payload: dict | None = None,
) -> MagicMock:
    """mock graph 暂停在 next_node。

    `interrupt_payload`:模拟节点内 `interrupt(payload)` 时,LangGraph 把 payload 存在
    snapshot.tasks[0].interrupts[0].value 的行为(diagnosis.py 的 initial_ask 分支
    会从这里取 followup_questions —— 因为节点 return 还没执行,state 未 commit)。
    """
    g = MagicMock()
    g.astream_events = _make_astream_events(
        events or _node_start_events(["info_collect"])
    )
    if interrupt_payload is not None:
        mock_interrupt = MagicMock(value=interrupt_payload)
        mock_task = MagicMock(interrupts=[mock_interrupt])
        snapshot = MagicMock(values=state_dict, next=(next_node,), tasks=[mock_task])
    else:
        snapshot = MagicMock(values=state_dict, next=(next_node,), tasks=[])
    g.aget_state = AsyncMock(return_value=snapshot)
    return g


def _read_sse_events(resp) -> list[dict]:
    """把 SSE 流响应正文切回 event dict 列表。

    SSE 格式:`data: <json>\\n\\n`(空行分隔消息)。本 helper 只取 `data:` 行的 JSON;
    忽略 event/id/retry 等其它 SSE 字段(本服务没用)。
    """
    events: list[dict] = []
    for raw_msg in resp.text.split("\n\n"):
        for line in raw_msg.split("\n"):
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


def _terminal_event(events: list[dict]) -> dict:
    """从 SSE event 序列里取最后一条非 progress(= interrupt/completed/error 之一)。"""
    for evt in reversed(events):
        if evt.get("event") in ("interrupt", "completed", "error"):
            return evt
    raise AssertionError(f"no terminal event in SSE stream; got: {events}")


# ────────────────────────────────────────────────────────────────────────────
# 鉴权 / 输入校验
# ────────────────────────────────────────────────────────────────────────────


def test_diagnose_without_token_returns_401(client: TestClient) -> None:
    resp = client.post("/diagnose", json={"patient_input": "腹痛三天"})
    assert resp.status_code == 401


def test_first_round_without_patient_input_returns_422(
    client: TestClient, patient_token
) -> None:
    """首次问诊必须带 patient_input(无 session_id 时)。"""
    token, _, _ = patient_token
    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"session_id": None},
    )
    assert resp.status_code == 422


# ────────────────────────────────────────────────────────────────────────────
# 终态 happy path + rag_trace 落库
# ────────────────────────────────────────────────────────────────────────────


def test_first_round_completed_writes_rag_trace(
    client: TestClient, patient_token, monkeypatch
) -> None:
    """模拟 graph 立刻终态 → 验响应 + DB 三张表(sessions / rag_trace / conversations)。"""
    from src.db.postgres.connection import session_scope

    token, user_id, _ = patient_token

    final_state = {
        "patient_input": "腹痛三天",
        "chief_complaint": "腹痛三天",
        "confirmed_symptoms": ["腹痛", "反酸"],
        "denied_symptoms": [],
        # standardized_entities 字段已随 EL 移除一并删除
        "candidate_chunks": [{"source_chunk_id": "c1", "rrf_score": 0.9}],
        "last_reranked_chunks": [{"source_chunk_id": "c1", "rerank_score": 0.92}],
        "last_diagnose_prompt": None,
        "last_diagnose_raw_output": None,
        "final_response": "建议查胃镜",
        "diagnosis_result": [
            {
                "disease": "胃炎",
                "probability": 0.7,
                "evidence": ["..."],
                "differentiation": None,
                "differentiation_type": "confirmed",
                "failure_reason": None,
            }
        ],
        "medication_advice": [{"drug": "奥美拉唑", "dosage": "20mg qd"}],
        "risk_warnings": ["如出现呕血请急诊"],
        "session_token_usage": {
            "prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380
        },
        "session_latency_ms": {
            "intent": 100, "retrieval": 200, "rerank": 50,
            "llm_call": 1500, "post_process": 30
        },
    }
    monkeypatch.setattr(
        "src.api.routes.diagnosis._get_compiled_graph",
        lambda: _mock_graph_completed(final_state),
    )

    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"patient_input": "腹痛三天"},
    )
    assert resp.status_code == 200, resp.text
    events = _read_sse_events(resp)
    terminal = _terminal_event(events)
    assert terminal["event"] == "completed"
    assert terminal["status"] == "completed"
    assert terminal["session_id"]
    assert terminal["final_response"] == "建议查胃镜"
    assert terminal["diagnosis_result"][0]["disease"] == "胃炎"
    assert terminal["risk_warnings"] == ["如出现呕血请急诊"]

    sid = terminal["session_id"]
    with session_scope() as s:
        # sessions 行
        cnt = s.execute(
            text("SELECT count(*) FROM sessions WHERE id = :sid"), {"sid": sid}
        ).scalar_one()
        assert cnt == 1

        # rag_trace 一行,15 字段就位
        trace = s.execute(
            text(
                "SELECT raw_query, intent_result, retrieved_chunks, "
                "reranked_chunks, final_response, model_name, token_usage, "
                "latency_ms, error_info, final_prompt FROM rag_trace "
                "WHERE session_id = :sid"
            ),
            {"sid": sid},
        ).one()
        assert trace[0] == "腹痛三天"
        assert trace[1]["chief_complaint"] == "腹痛三天"
        assert trace[1]["confirmed_symptoms"] == ["腹痛", "反酸"]
        assert trace[2][0]["source_chunk_id"] == "c1"
        assert trace[3][0]["rerank_score"] == 0.92
        assert trace[4] == "建议查胃镜"
        assert trace[5]  # model_name 非空
        assert trace[6]["total_tokens"] == 1380
        assert trace[7]["total"] >= 0  # invoke_latency_ms
        assert trace[8] is None  # 正常路径 error_info NULL
        assert trace[9] is None  # final_prompt 正常路径 NULL

        # conversations 一行
        conv = s.execute(
            text(
                "SELECT user_input, llm_output, rag_context FROM conversations "
                "WHERE session_id = :sid"
            ),
            {"sid": sid},
        ).one()
        assert conv[0] == "腹痛三天"
        assert conv[1] == "建议查胃镜"
        assert conv[2]["chunk_ids"] == ["c1"]


def test_failure_path_writes_error_info(
    client: TestClient, patient_token, monkeypatch
) -> None:
    """diagnose ⑩ 失败兜底场景:diagnosis_result[0].failure_reason → error_info."""
    from src.db.postgres.connection import session_scope

    token, user_id, _ = patient_token
    final_state = {
        "patient_input": "x",
        "chief_complaint": "",
        "confirmed_symptoms": [],
        "denied_symptoms": [],
        # standardized_entities 字段已随 EL 移除一并删除
        "candidate_chunks": [],
        "last_reranked_chunks": [],
        "last_diagnose_prompt": "<step 2 prompt>",
        "last_diagnose_raw_output": "<malformed json>",
        "final_response": "信息不足以支持可靠诊断",
        "diagnosis_result": [
            {
                "disease": "信息不足以支持可靠诊断",
                "probability": 0.0,
                "failure_reason": "step_2_structured_output_failed: ValidationError: x",
            }
        ],
        "medication_advice": [],
        "risk_warnings": [],
        "session_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "session_latency_ms": {"intent": 0, "retrieval": 0, "rerank": 0, "llm_call": 0, "post_process": 0},
    }
    monkeypatch.setattr(
        "src.api.routes.diagnosis._get_compiled_graph",
        lambda: _mock_graph_completed(final_state),
    )

    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"patient_input": "x"},
    )
    assert resp.status_code == 200
    sid = _terminal_event(_read_sse_events(resp))["session_id"]

    with session_scope() as s:
        row = s.execute(
            text(
                "SELECT error_info, final_prompt, llm_raw_output FROM rag_trace "
                "WHERE session_id = :sid"
            ),
            {"sid": sid},
        ).one()
        assert row[0]["step"] == 2
        assert row[0]["failure_reason"].startswith("step_2_structured_output_failed")
        assert row[1] == "<step 2 prompt>"
        assert row[2] == "<malformed json>"


# ────────────────────────────────────────────────────────────────────────────
# interrupt 状态机
# ────────────────────────────────────────────────────────────────────────────


def test_first_round_interrupt_returns_ongoing_followup(
    client: TestClient, patient_token, monkeypatch
) -> None:
    """graph 暂停在 wait_followup_answer → status=ongoing_followup + pending_question."""
    token, _, _ = patient_token
    state = {
        "patient_input": "胃疼",
        "followup_question": "疼痛多久了?",
    }
    monkeypatch.setattr(
        "src.api.routes.diagnosis._get_compiled_graph",
        lambda: _mock_graph_interrupt(state, "wait_followup_answer"),
    )

    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"patient_input": "胃疼"},
    )
    assert resp.status_code == 200
    terminal = _terminal_event(_read_sse_events(resp))
    assert terminal["event"] == "interrupt"
    assert terminal["status"] == "ongoing_followup"
    assert terminal["pending_question"] == "疼痛多久了?"
    assert terminal["session_id"]


def test_interrupt_at_wait_exam_report_returns_ongoing_exam(
    client: TestClient, patient_token, monkeypatch
) -> None:
    """graph 暂停在 wait_exam_report → status=ongoing_exam + recommended_tests."""
    token, _, _ = patient_token
    state = {
        "patient_input": "胃疼",
        "recommended_tests": ["胃镜", "幽门螺杆菌检测"],
    }
    monkeypatch.setattr(
        "src.api.routes.diagnosis._get_compiled_graph",
        lambda: _mock_graph_interrupt(state, "wait_exam_report"),
    )

    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"patient_input": "胃疼"},
    )
    assert resp.status_code == 200
    terminal = _terminal_event(_read_sse_events(resp))
    assert terminal["event"] == "interrupt"
    assert terminal["status"] == "ongoing_exam"
    assert terminal["recommended_tests"] == ["胃镜", "幽门螺杆菌检测"]


def test_interrupt_at_initial_ask_returns_initial_form(
    client: TestClient, patient_token, monkeypatch
) -> None:
    """⓪a 节点内 interrupt → status=ongoing_initial_ask + pending_questions 含 open/history/(obstetric)。"""
    token, _, _ = patient_token
    state = {"patient_input": "腹痛"}  # ⓪a 入口 interrupt,state 还没 commit
    interrupt_payload = {
        "followup_question": "您还有其他不适吗?\n\n您有过敏/慢病/长期用药吗?",
        "followup_questions": [
            {"type": "open", "question": "您还有其他不适吗?"},
            {"type": "history", "question": "您有过敏/慢病/长期用药吗?"},
        ],
    }
    monkeypatch.setattr(
        "src.api.routes.diagnosis._get_compiled_graph",
        lambda: _mock_graph_interrupt(
            state, "initial_ask", interrupt_payload=interrupt_payload
        ),
    )

    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"patient_input": "腹痛"},
    )
    assert resp.status_code == 200
    terminal = _terminal_event(_read_sse_events(resp))
    assert terminal["event"] == "interrupt"
    assert terminal["status"] == "ongoing_initial_ask"
    pq = terminal["pending_questions"]
    types = {q["type"] for q in pq}
    assert "open" in types
    assert "history" in types


def test_interrupt_at_analyze_initial_reports_returns_report_upload(
    client: TestClient, patient_token, monkeypatch
) -> None:
    """①.5 节点内 interrupt → status=ongoing_report_upload + pending_questions 含 report_upload 项。"""
    token, _, _ = patient_token
    state = {"patient_input": "腹痛"}
    monkeypatch.setattr(
        "src.api.routes.diagnosis._get_compiled_graph",
        lambda: _mock_graph_interrupt(state, "analyze_initial_reports"),
    )

    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"patient_input": "腹痛"},
    )
    assert resp.status_code == 200
    terminal = _terminal_event(_read_sse_events(resp))
    assert terminal["event"] == "interrupt"
    assert terminal["status"] == "ongoing_report_upload"
    pq = terminal["pending_questions"]
    assert len(pq) == 1
    assert pq[0]["type"] == "report_upload"
    assert "检查报告" in pq[0]["question"]


# ────────────────────────────────────────────────────────────────────────────
# session 越权 / 不存在
# ────────────────────────────────────────────────────────────────────────────


def test_unknown_session_id_returns_404(
    client: TestClient, patient_token
) -> None:
    token, _, _ = patient_token
    resp = client.post(
        "/diagnose",
        headers={"Authorization": f"Bearer {token}"},
        json={"session_id": str(uuid.uuid4()), "followup_answer": "x"},
    )
    assert resp.status_code == 404


def test_session_owned_by_other_user_returns_403(
    client: TestClient, patient_token, monkeypatch
) -> None:
    """A 用户的 session_id,B 用户拿来跑 → 403。"""
    from src.db.postgres.connection import session_scope
    from src.db.postgres.models_dialog import Session as SessionRow
    from src.db.postgres.models_patient import User
    from src.api.middleware.auth_middleware import hash_password

    _, user_a, _ = patient_token

    # 建一个 B 用户 + 一个属于 B 的 session
    email_b = f"g4_other_{uuid.uuid4().hex[:8]}@example.com"
    with session_scope() as s:
        u_b = User(email=email_b, password=hash_password("x"), role="patient")
        s.add(u_b)
        s.flush()
        sess_b = SessionRow(user_id=u_b.id, title="B 的会话")
        s.add(sess_b)
        s.flush()
        sess_b_id = str(sess_b.id)
        user_b_id = str(u_b.id)

    try:
        # A 用 token 访问 B 的 session_id
        resp = client.post(
            "/diagnose",
            headers={"Authorization": f"Bearer {patient_token[0]}"},
            json={"session_id": sess_b_id, "followup_answer": "x"},
        )
        assert resp.status_code == 403
    finally:
        with session_scope() as s:
            # sessions / rag_trace / conversations FK→users 不级联删,手动先清
            s.execute(text("DELETE FROM rag_trace WHERE user_id = :uid"), {"uid": user_b_id})
            s.execute(text("DELETE FROM conversations WHERE user_id = :uid"), {"uid": user_b_id})
            s.execute(text("DELETE FROM sessions WHERE user_id = :uid"), {"uid": user_b_id})
            s.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_b_id})
