"""tests/integration/test_diagnosis_routes.py — G4 POST /diagnose 闭环。

graph 用 mock 替换(真跑会调 LLM/Embedding/Reranker,慢 + 烧 token)。
mock 模拟三种 graph 形态:
- 首轮立刻终态 → status="completed" + rag_trace + conversation 写入
- 首轮 interrupt → status="ongoing_followup" + pending_question
- resume 后终态 → 同上 + rag_trace 写入

需 PG 真服务在跑 + alembic upgrade head。
"""
from __future__ import annotations

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


def _mock_graph_completed(final_state: dict) -> MagicMock:
    """mock graph 立刻终态:ainvoke 返 final_state,aget_state next 空。"""
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value=final_state)
    snapshot = MagicMock(values=final_state, next=())
    g.aget_state = AsyncMock(return_value=snapshot)
    return g


def _mock_graph_interrupt(state_dict: dict, next_node: str) -> MagicMock:
    """mock graph 暂停在 next_node。"""
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value=state_dict)
    snapshot = MagicMock(values=state_dict, next=(next_node,))
    g.aget_state = AsyncMock(return_value=snapshot)
    return g


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
    body = resp.json()
    assert body["status"] == "completed"
    assert body["session_id"]
    assert body["final_response"] == "建议查胃镜"
    assert body["diagnosis_result"][0]["disease"] == "胃炎"
    assert body["risk_warnings"] == ["如出现呕血请急诊"]

    sid = body["session_id"]
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
    sid = resp.json()["session_id"]

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
    body = resp.json()
    assert body["status"] == "ongoing_followup"
    assert body["pending_question"] == "疼痛多久了?"
    assert body["session_id"]


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
    body = resp.json()
    assert body["status"] == "ongoing_exam"
    assert body["recommended_tests"] == ["胃镜", "幽门螺杆菌检测"]


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
