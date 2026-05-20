"""创建 3 个 demo 患者账号 + 预置档案,供 ui/ 界面演示用。

幂等:已存在 email 的账号 → 跳过(不更新档案,避免覆盖手动编辑)。
密码统一 `demo1234`,只用于本地 / portfolio 演示,**禁止用在任何生产环境**。

3 个 persona:
  patient1@demo.com — 张女士 32 岁 孕 16 周 → 演示 ⑪ safety_gate 妊娠禁忌兜底
  patient2@demo.com — 李先生 65 岁 高血压+糖尿病+青霉素过敏 → 演示 ⑩ 拿病史推理 + ⑪ 过敏拦截
  patient3@demo.com — 王同学 25 岁 完全空档案 → 演示首诊新患者基线

用法:
    python scripts/init_demo_users.py            # 创建缺失账号
    python scripts/init_demo_users.py --force    # 已有也重写(危险:清空档案重灌)
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

from sqlalchemy import delete, select

from src.api.middleware.auth_middleware import hash_password
from src.db.postgres.connection import session_scope
from src.db.postgres.models_patient import (
    Allergy,
    MedicalHistory,
    Medication,
    MenstrualReproductive,
    Patient,
    User,
)


_logger = logging.getLogger(__name__)

PASSWORD = "demo1234"


# ────────────────────────────────────────────────────────────────────────────
# Demo 数据
# ────────────────────────────────────────────────────────────────────────────


DEMO_USERS = [
    {
        "email": "patient1@demo.com",
        "patient": {
            "name": "张女士",
            "gender": "female",
            "birth_date": date(1992, 1, 1),
            "height_cm": 162,
            "blood_type": "A+",
        },
        "obstetric": {
            "is_pregnant": True,
            "is_lactating": False,
            "gravidity": 1,
            "parity": 0,
            "menarche_age": 13,
            "last_menstrual_period": date(2026, 1, 25),
        },
    },
    {
        "email": "patient2@demo.com",
        "patient": {
            "name": "李先生",
            "gender": "male",
            "birth_date": date(1960, 6, 15),
            "height_cm": 170,
            "blood_type": "O+",
            "smoking_status": "former",
            "smoking_pack_years": 20,
            "alcohol_status": "occasional",
        },
        "medical_history": [
            {
                "category": "chronic",
                "condition": "高血压",
                "diagnosed_at": date(2021, 3, 1),
                "control_status": "well_controlled",
                "notes": "服药后控制良好",
            },
            {
                "category": "chronic",
                "condition": "2型糖尿病",
                "diagnosed_at": date(2022, 9, 1),
                "control_status": "well_controlled",
            },
        ],
        "medications": [
            {
                "drug_name": "氨氯地平",
                "drug_category": "antihypertensive",
                "dosage": "5mg",
                "frequency": "每日一次",
                "route": "oral",
                "started_at": date(2021, 3, 15),
                "is_self_medication": False,
            },
            {
                "drug_name": "二甲双胍",
                "drug_category": "hypoglycemic",
                "dosage": "500mg",
                "frequency": "每日两次",
                "route": "oral",
                "started_at": date(2022, 9, 10),
                "is_self_medication": False,
            },
        ],
        "allergies": [
            {
                "allergen": "青霉素",
                "allergen_type": "drug",
                "reaction": "全身荨麻疹伴呼吸困难",
                "severity": "severe",
                "status": "confirmed",
            }
        ],
    },
    {
        "email": "patient3@demo.com",
        "patient": {
            "name": "王同学",
            "gender": "male",
            "birth_date": date(2000, 8, 20),
            "height_cm": 178,
        },
        # 完全空档案,演示首诊新患者基线
    },
]


# ────────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────────


def _build_user_with_archive(session, spec: dict) -> str:
    """同一事务中创建 User + Patient + 子表。返回 user_id。"""
    user = User(
        email=spec["email"],
        password=hash_password(PASSWORD),
        role="patient",
    )
    session.add(user)
    session.flush()  # 拿 id

    patient = Patient(id=user.id, **spec["patient"])
    session.add(patient)
    session.flush()

    for mh in spec.get("medical_history", []):
        session.add(MedicalHistory(patient_id=user.id, **mh))
    for med in spec.get("medications", []):
        session.add(Medication(patient_id=user.id, **med))
    for al in spec.get("allergies", []):
        session.add(Allergy(patient_id=user.id, **al))
    if "obstetric" in spec:
        session.add(MenstrualReproductive(patient_id=user.id, **spec["obstetric"]))

    return user.id


def _delete_user_and_archive(session, user_id: str) -> None:
    """级联删:patients 外键 ON DELETE CASCADE 已覆盖子表;User 删了即可。"""
    session.execute(delete(User).where(User.id == user_id))


def main() -> int:
    parser = argparse.ArgumentParser(description="创建 3 个 demo 患者账号")
    parser.add_argument("--force", action="store_true", help="已存在也重灌(先删后建)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    created = 0
    skipped = 0
    rewrote = 0

    for spec in DEMO_USERS:
        email = spec["email"]
        with session_scope() as s:
            existing = s.execute(
                select(User).where(User.email == email)
            ).scalar_one_or_none()

            if existing is not None:
                if not args.force:
                    _logger.info("[skip] %s 已存在(--force 可重灌)", email)
                    skipped += 1
                    continue
                _logger.info("[force] 重灌 %s", email)
                _delete_user_and_archive(s, existing.id)
                s.flush()
                _build_user_with_archive(s, spec)
                rewrote += 1
            else:
                _build_user_with_archive(s, spec)
                created += 1
                _logger.info("[ok] 创建 %s (密码: %s)", email, PASSWORD)

    print(f"\n=== 汇总 ===")
    print(f"  新建: {created}")
    print(f"  跳过(已存在): {skipped}")
    print(f"  重灌(--force): {rewrote}")
    if created or rewrote:
        print(f"\n  统一密码: {PASSWORD}")
        print(f"  登录入口: http://localhost:8000/ui (启动 uvicorn 后)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
