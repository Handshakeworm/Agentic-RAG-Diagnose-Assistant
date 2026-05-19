"""tests/integration/test_dialog_audit_crud.py — 真起 PG 验证 §2.4.3 + §5.2.3 + §5.3。

覆盖:
- sessions / conversations FK 链 + active partial index
- rag_trace 15 字段一次性 insert(模拟 G4 endpoint 写入路径,§9.6.5)
- diagnosis_feedback 关联 trace_id 且按 rating 索引查询
- system_config upsert + 与 config_change_log 的同事务写入

需要 PG 真服务 + 0001~0006 全部迁移已应用(或 alembic upgrade head)。
"""
from __future__ import annotations

import os
import socket
import uuid

import pytest
from sqlalchemy import text


def _pg_alive() -> bool:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _pg_alive(), reason="PG 不可达,启动 docker compose 后再跑")


# ────────────────────────────────────────────────────────────────────────────
# fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def user_and_session():
    """建一个 user + 一个 active session,teardown 清理。"""
    from src.db.postgres.connection import session_scope
    from src.db.postgres.models_dialog import Session as SessionRow
    from src.db.postgres.models_patient import User

    email = f"dialog_{uuid.uuid4().hex[:8]}@example.com"
    with session_scope() as s:
        u = User(email=email, password="x", role="patient")
        s.add(u)
        s.flush()
        s.refresh(u)
        user_id = u.id

        sess = SessionRow(user_id=user_id, title="测试问诊")
        s.add(sess)
        s.flush()
        s.refresh(sess)
        session_id = sess.id

    yield user_id, session_id

    with session_scope() as s:
        # 顺序:diagnosis_feedback → rag_trace → conversations → sessions → users
        s.execute(text("DELETE FROM diagnosis_feedback WHERE reviewer_id = :uid"), {"uid": user_id})
        s.execute(text("DELETE FROM rag_trace WHERE user_id = :uid"), {"uid": user_id})
        s.execute(text("DELETE FROM conversations WHERE user_id = :uid"), {"uid": user_id})
        s.execute(text("DELETE FROM sessions WHERE user_id = :uid"), {"uid": user_id})
        s.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})


# ────────────────────────────────────────────────────────────────────────────
# §2.4.3 sessions / conversations
# ────────────────────────────────────────────────────────────────────────────


def test_conversation_with_rag_context_jsonb(user_and_session) -> None:
    """conversations.rag_context 存检索快照(chunk_ids + scores)。"""
    from src.db.postgres.connection import session_scope
    from src.db.postgres.models_dialog import Conversation

    user_id, session_id = user_and_session
    snapshot = {
        "chunk_ids": ["chunk_a", "chunk_b"],
        "scores": [0.92, 0.81],
    }
    with session_scope() as s:
        s.add(
            Conversation(
                session_id=session_id,
                user_id=user_id,
                user_input="腹痛三天",
                llm_output="建议查消化系统",
                rag_context=snapshot,
            )
        )

    with session_scope() as s:
        row = s.execute(
            text("SELECT rag_context FROM conversations WHERE session_id = :sid"),
            {"sid": session_id},
        ).scalar_one()
        assert row["chunk_ids"] == snapshot["chunk_ids"]
        assert row["scores"] == snapshot["scores"]


def test_active_session_partial_index(user_and_session) -> None:
    """spec §2.4.3:idx_sessions_status_active 只覆盖 status='active' 行。"""
    from src.db.postgres.connection import session_scope

    user_id, _ = user_and_session
    with session_scope() as s:
        rows = s.execute(
            text("SELECT count(*) FROM sessions WHERE user_id = :uid AND status = 'active'"),
            {"uid": user_id},
        ).scalar_one()
        assert rows == 1


# ────────────────────────────────────────────────────────────────────────────
# §5.2.3.1 rag_trace — 15 字段一次性写入(模拟 §9.6.5 G4 视图函数)
# ────────────────────────────────────────────────────────────────────────────


def test_rag_trace_full_15_field_insert(user_and_session) -> None:
    """模拟 G4 endpoint 按 §9.6.5 裸代码模板组装并写入 rag_trace。
    验证 15 字段全部能落库 + JSONB 字段往返一致。"""
    from src.db.postgres.connection import session_scope
    from src.db.postgres.models_audit import RagTrace

    user_id, session_id = user_and_session
    payload = RagTrace(
        session_id=session_id,
        user_id=user_id,
        raw_query="腹痛 3 天,进食后加重",
        intent_result={
            "chief_complaint": "腹痛 3 天",
            "confirmed_symptoms": ["反酸", "进食后加重"],
            "denied_symptoms": [],
            # standardized_entities 字段已随 EL 移除一并删除
        },
        retrieved_chunks=[{"chunk_id": "c1", "rrf_score": 0.95}],
        reranked_chunks=[{"chunk_id": "c1", "rerank_score": 0.92, "rank": 1}],
        final_prompt=None,  # 正常路径 NULL(spec §9.6.2)
        llm_raw_output=None,
        final_response="建议查胃镜",
        model_name="deepseek-v4-pro",
        token_usage={"prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380},
        latency_ms={"intent": 450, "retrieval": 320, "rerank": 80, "llm_call": 2400, "total": 3250},
        error_info=None,
    )
    with session_scope() as s:
        s.add(payload)
        s.flush()
        s.refresh(payload)
        trace_id = payload.trace_id

    with session_scope() as s:
        row = s.execute(
            text(
                "SELECT raw_query, intent_result, token_usage, model_name, "
                "final_prompt, error_info FROM rag_trace WHERE trace_id = :tid"
            ),
            {"tid": trace_id},
        ).one()
        assert row[0] == "腹痛 3 天,进食后加重"
        assert row[1]["chief_complaint"] == "腹痛 3 天"
        assert row[2]["total_tokens"] == 1380
        assert row[3] == "deepseek-v4-pro"
        assert row[4] is None  # 正常路径 final_prompt 为 NULL
        assert row[5] is None


def test_rag_trace_failure_path_fills_prompt_and_raw(user_and_session) -> None:
    """⑩ diagnose 失败兜底路径:final_prompt + llm_raw_output + error_info 三字段填值。"""
    from src.db.postgres.connection import session_scope
    from src.db.postgres.models_audit import RagTrace

    user_id, session_id = user_and_session
    err = {
        "source": "diagnose",
        "failure_reason": "step_2_structured_output_failed: ValidationError: ...",
        "step": 2,
    }
    with session_scope() as s:
        s.add(
            RagTrace(
                session_id=session_id,
                user_id=user_id,
                raw_query="发热 5 天",
                final_response="信息不足以支持可靠诊断",
                model_name="deepseek-v4-pro",
                final_prompt="<step 2 ranking prompt full text>",
                llm_raw_output='{"diseases": [<malformed JSON>',
                error_info=err,
            )
        )

    with session_scope() as s:
        row = s.execute(
            text(
                "SELECT error_info, final_prompt FROM rag_trace "
                "WHERE user_id = :uid AND error_info IS NOT NULL"
            ),
            {"uid": user_id},
        ).one()
        assert row[0]["step"] == 2
        assert row[0]["failure_reason"].startswith("step_2_structured_output_failed")
        assert row[1].startswith("<step 2 ranking prompt")


# ────────────────────────────────────────────────────────────────────────────
# §5.2.3.4 diagnosis_feedback — 关联 trace_id 标注
# ────────────────────────────────────────────────────────────────────────────


def test_feedback_links_to_trace(user_and_session) -> None:
    """spec §5.2.3.4 + §5.2.3.5 联合分析:trace 标注后能通过 trace_id 回溯。"""
    from src.db.postgres.connection import session_scope
    from src.db.postgres.models_audit import DiagnosisFeedback, RagTrace

    user_id, session_id = user_and_session
    with session_scope() as s:
        trace = RagTrace(
            session_id=session_id,
            user_id=user_id,
            raw_query="x",
            final_response="y",
            model_name="deepseek-v4-pro",
        )
        s.add(trace)
        s.flush()
        s.refresh(trace)
        trace_id = trace.trace_id

        s.add(
            DiagnosisFeedback(
                trace_id=trace_id,
                reviewer_id=user_id,  # 同一个 user 既看诊也标注
                rating="HALLUCINATION",
                comment="检索到正确文档但 LLM 理解错误",
                expected_response="应该建议做胃镜而不是CT",
            )
        )

    with session_scope() as s:
        cnt = s.execute(
            text(
                "SELECT count(*) FROM diagnosis_feedback "
                "WHERE rating = 'HALLUCINATION' AND trace_id = :tid"
            ),
            {"tid": trace_id},
        ).scalar_one()
        assert cnt == 1


# ────────────────────────────────────────────────────────────────────────────
# §5.3 system_config — 同事务写 system_config + config_change_log
# ────────────────────────────────────────────────────────────────────────────


def test_system_config_upsert_with_change_log(user_and_session) -> None:
    """spec §5.3 末:admin 修改 system_config 时同事务写 config_change_log。"""
    from src.db.postgres.connection import session_scope
    from src.db.postgres.models_audit import ConfigChangeLog, SystemConfig

    user_id, _ = user_and_session
    key = f"test_{uuid.uuid4().hex[:8]}"

    try:
        # 初次写入
        with session_scope() as s:
            s.add(
                SystemConfig(
                    key_name=key,
                    value=0.7,
                    value_type="FLOAT",
                    description="test",
                    updated_by=user_id,
                )
            )

        # admin 改值 + 同事务写 change_log(模拟 G6 admin endpoint 视图)
        with session_scope() as s:
            cfg = s.get(SystemConfig, key)
            old_val = cfg.value
            cfg.value = 0.3
            cfg.updated_by = user_id
            s.add(
                ConfigChangeLog(
                    operator_id=user_id,
                    config_key=key,
                    old_value=old_val,
                    new_value=0.3,
                    change_reason="降低温度以减少幻觉",
                )
            )

        # 验证
        with session_scope() as s:
            cur = s.get(SystemConfig, key)
            assert float(cur.value) == 0.3
            log = s.execute(
                text(
                    "SELECT old_value, new_value FROM config_change_log "
                    "WHERE config_key = :k"
                ),
                {"k": key},
            ).one()
            assert float(log[0]) == 0.7
            assert float(log[1]) == 0.3
    finally:
        with session_scope() as s:
            s.execute(text("DELETE FROM config_change_log WHERE config_key = :k"), {"k": key})
            s.execute(text("DELETE FROM system_config WHERE key_name = :k"), {"k": key})
