"""tests/integration/test_patient_routes.py — G5 患者档案端点闭环。

覆盖:
- 角色守卫:admin token 访问 /patients/me 应 403
- GET /patients/me 主档案 + 8 张子表汇总
- PUT /patients/me 基本信息更新(不存在则自动建 patients 行)
- 三张 ⚠️必问表 POST/DELETE 闭环
- 身份隔离:DELETE 别人的子表行 → 404
"""
from __future__ import annotations

import os
import socket
import uuid

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


@pytest.fixture
def client() -> TestClient:
    from src.api.app import app
    return TestClient(app)


def _register(client: TestClient, role: str = "patient") -> tuple[str, str]:
    email = f"g5_{role}_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/auth/register",
        json={"email": email, "password": "hunter22", "role": role},
    )
    assert resp.status_code == 201
    return resp.json()["access_token"], email


@pytest.fixture
def patient_session():
    """注册 patient + 自动 teardown(级联清 patients/子表 + users)。"""
    from src.api.app import app
    from src.db.postgres.connection import session_scope

    with TestClient(app) as c:
        token, email = _register(c)
    yield token, email
    with session_scope() as s:
        s.execute(text("DELETE FROM users WHERE email = :e"), {"e": email})


# ────────────────────────────────────────────────────────────────────────────
# 角色守卫
# ────────────────────────────────────────────────────────────────────────────


def test_admin_role_cannot_access_patient_endpoints(client: TestClient) -> None:
    """admin 角色访问 /patients/me 应 403(spec G5:仅 patient 角色)。"""
    from src.db.postgres.connection import session_scope

    token, email = _register(client, role="admin")
    try:
        resp = client.get("/patients/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403
    finally:
        with session_scope() as s:
            s.execute(text("DELETE FROM users WHERE email = :e"), {"e": email})


# ────────────────────────────────────────────────────────────────────────────
# GET / PUT 主档案
# ────────────────────────────────────────────────────────────────────────────


def test_get_profile_for_new_user_returns_empty_history(
    client: TestClient, patient_session
) -> None:
    """新注册用户没填档案,8 张子表都返空。"""
    token, email = patient_session
    resp = client.get("/patients/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == email
    assert body["name"] is None  # patients 行不存在 → 字段全 None
    assert body["allergy_history"] == []
    assert body["medication_history"] == []
    assert body["family_history"] == []


def test_put_profile_creates_patients_row_on_first_update(
    client: TestClient, patient_session
) -> None:
    """spec §2.4.5:patients 1:1 与 users。注册时只建 users,patients 延后到首次填档。"""
    token, _ = patient_session
    resp = client.put(
        "/patients/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "测试用户",
            "gender": "male",
            "birth_date": "1990-01-01",
            "smoking_status": "never",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "测试用户"
    assert body["gender"] == "male"
    assert body["birth_date"] == "1990-01-01"
    assert body["personal_history"]["smoking_status"] == "never"


def test_put_profile_only_updates_provided_fields(
    client: TestClient, patient_session
) -> None:
    """exclude_unset 让 PUT 只动显式给的字段(部分更新语义)。"""
    token, _ = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    client.put("/patients/me", headers=headers, json={"name": "甲", "gender": "male"})
    client.put("/patients/me", headers=headers, json={"name": "乙"})  # 只改 name

    body = client.get("/patients/me", headers=headers).json()
    assert body["name"] == "乙"
    assert body["gender"] == "male"  # 未传 → 不变


# ────────────────────────────────────────────────────────────────────────────
# 三张 ⚠️必问表 POST/DELETE
# ────────────────────────────────────────────────────────────────────────────


def test_create_and_delete_allergy(client: TestClient, patient_session) -> None:
    token, _ = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    # POST
    resp = client.post(
        "/patients/me/allergies",
        headers=headers,
        json={
            "allergen": "青霉素",
            "allergen_type": "drug",
            "severity": "severe",
            "status": "confirmed",
        },
    )
    assert resp.status_code == 201
    record_id = resp.json()["id"]

    # GET 主档案 → 含此条
    body = client.get("/patients/me", headers=headers).json()
    assert any(a["substance"] == "青霉素" for a in body["allergy_history"])

    # DELETE → 204
    resp = client.delete(f"/patients/me/allergies/{record_id}", headers=headers)
    assert resp.status_code == 204

    # 再 GET → 已没有
    body = client.get("/patients/me", headers=headers).json()
    assert body["allergy_history"] == []


def test_create_medication_and_visible_via_safety_gate_query(
    client: TestClient, patient_session
) -> None:
    """safety_gate ⑪ 读 medications WHERE patient_id=... — 验整链能走通。"""
    from src.agent.utils.patient_repo import load_medical_history
    from src.db.postgres.connection import session_scope

    token, email = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    client.post(
        "/patients/me/medications",
        headers=headers,
        json={
            "drug_name": "二甲双胍",
            "drug_category": "hypoglycemic",
            "dosage": "500mg",
            "frequency": "每日两次",
            "is_self_medication": False,
        },
    )

    # 直接调 patient_repo(safety_gate 用的同一接口)
    with session_scope() as s:
        user_id = s.execute(
            text("SELECT id FROM users WHERE email = :e"), {"e": email}
        ).scalar_one()
    history = load_medical_history(str(user_id))
    assert any(m["drug_name"] == "二甲双胍" for m in history["medication_history"])


def test_delete_other_users_record_returns_404(client: TestClient) -> None:
    """A 拿 B 的 medication record_id 调 DELETE → 404(防泄漏 + 防越权)。"""
    from src.db.postgres.connection import session_scope

    token_a, email_a = _register(client)
    token_b, email_b = _register(client)

    try:
        # B 创建一条用药
        resp = client.post(
            "/patients/me/medications",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"drug_name": "胰岛素"},
        )
        record_id = resp.json()["id"]

        # A 尝试删 → 404
        resp = client.delete(
            f"/patients/me/medications/{record_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert resp.status_code == 404

        # B 自己删可以
        resp = client.delete(
            f"/patients/me/medications/{record_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert resp.status_code == 204
    finally:
        with session_scope() as s:
            s.execute(text("DELETE FROM users WHERE email IN (:a, :b)"), {"a": email_a, "b": email_b})


def test_create_medical_history_and_get_back(client: TestClient, patient_session) -> None:
    token, _ = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/patients/me/medical-history",
        headers=headers,
        json={
            "category": "chronic",
            "condition": "2型糖尿病",
            "icd10_code": "E11",
            "control_status": "well_controlled",
        },
    )
    assert resp.status_code == 201

    body = client.get("/patients/me", headers=headers).json()
    items = body["past_history"]["medical_history"]
    assert len(items) == 1
    assert items[0]["condition"] == "2型糖尿病"
    assert items[0]["icd10_code"] == "E11"


# ────────────────────────────────────────────────────────────────────────────
# Obstetric (menstrual_reproductive) 1:1 子表 PUT upsert + DELETE
# 验 ⑪ safety_gate 妊娠/哺乳禁忌兜底硬依赖的写入路径
# ────────────────────────────────────────────────────────────────────────────


def test_obstetric_put_creates_and_get_returns(client: TestClient, patient_session) -> None:
    """首次 PUT → 创建新行;GET 主档案 → obstetric_history 含写入字段。"""
    token, _ = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.put(
        "/patients/me/obstetric",
        headers=headers,
        json={"is_pregnant": True, "gravidity": 1, "parity": 0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_pregnant"] is True
    assert body["gravidity"] == 1

    profile = client.get("/patients/me", headers=headers).json()
    ob = profile["obstetric_history"]
    assert ob["pregnancy_status"] == "pregnant"  # ⑪ safety_gate 读这字段


def test_obstetric_put_upsert_updates_existing(client: TestClient, patient_session) -> None:
    """二次 PUT 同患者 → 更新现有行,partial update 不清空未传字段。"""
    token, _ = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    # 首次:写 is_pregnant + gravidity
    client.put(
        "/patients/me/obstetric",
        headers=headers,
        json={"is_pregnant": True, "gravidity": 2},
    )
    # 二次:仅改 is_lactating(未传 is_pregnant/gravidity,应保留)
    resp = client.put(
        "/patients/me/obstetric",
        headers=headers,
        json={"is_lactating": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_pregnant"] is True  # 保留
    assert body["gravidity"] == 2  # 保留
    assert body["is_lactating"] is True  # 新增


def test_obstetric_delete_removes_row(client: TestClient, patient_session) -> None:
    """DELETE → 204;再 GET → obstetric_history 为 None。"""
    token, _ = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    client.put(
        "/patients/me/obstetric",
        headers=headers,
        json={"is_pregnant": False, "is_lactating": True},
    )
    resp = client.delete("/patients/me/obstetric", headers=headers)
    assert resp.status_code == 204

    profile = client.get("/patients/me", headers=headers).json()
    assert profile["obstetric_history"] is None


def test_obstetric_delete_when_no_row_is_idempotent(client: TestClient, patient_session) -> None:
    """从未写过 → DELETE 也返 204(幂等)。"""
    token, _ = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.delete("/patients/me/obstetric", headers=headers)
    assert resp.status_code == 204


def test_obstetric_load_medical_history_returns_basic_info(
    client: TestClient, patient_session
) -> None:
    """load_medical_history 返回顶层新增 basic_info(含 gender),供 ④ 强制问妊娠用。"""
    from src.agent.utils.patient_repo import load_medical_history
    from src.db.postgres.connection import session_scope

    token, email = patient_session
    headers = {"Authorization": f"Bearer {token}"}

    # 先 PUT 基本信息设 gender=female
    client.put(
        "/patients/me",
        headers=headers,
        json={"gender": "female", "name": "测试"},
    )

    with session_scope() as s:
        user_id = s.execute(
            text("SELECT id FROM users WHERE email = :e"), {"e": email}
        ).scalar_one()

    history = load_medical_history(str(user_id))
    assert "basic_info" in history
    assert history["basic_info"]["gender"] == "female"
    assert history["basic_info"]["name"] == "测试"
