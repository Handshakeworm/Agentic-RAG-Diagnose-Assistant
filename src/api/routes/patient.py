"""患者档案接口(DEV_SPEC §8.4 G5 / §2.4.5)。

挂在 `/patients/*`,**全部** require_role("patient") — admin 想看用户档案走 G6 接口。

身份隔离:全部端点用 `current_user.user_id`,**没有 path 参数** —
"我的档案" 永远只能动自己的,从设计上根除"传 path id 越权"的风险。

端点列表(10):
- GET    /patients/me                       全档案
- PUT    /patients/me                       基本信息 + 个人史
- POST   /patients/me/medical-history       新建一条既往病史
- DELETE /patients/me/medical-history/{id}  删除一条
- POST   /patients/me/allergies             新建一条过敏史
- DELETE /patients/me/allergies/{id}        删除一条
- POST   /patients/me/medications           新建一条用药
- DELETE /patients/me/medications/{id}      删除一条
- PUT    /patients/me/obstetric             婚育/月经史 upsert(1:1 子表)
- DELETE /patients/me/obstetric             删除整行

⚠️ TODO(留给用户拍板):
- 4 张子表(surgical_trauma / transfusion / family_history / exam_reports)
  只在 GET /patients/me 返回值里读出,独立 CRUD 端点本期未实现。是否补?
- exam_reports 还涉及文件上传,需要 multipart endpoint + 落盘 / 对象存储,
  这是一块单独工作量
- PUT 子表(改医生处方等)— 简化为"删了重建"足够 demo,真生产要 PUT
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from src.agent.utils.patient_repo import load_medical_history
from src.api.middleware.auth_middleware import CurrentUser, require_role
from src.api.routes.auth import get_db
from src.api.schemas.patient_schema import (
    AllergyCreate,
    CreatedRecordOut,
    MedicalHistoryCreate,
    MedicationCreate,
    ObstetricOut,
    ObstetricUpdate,
    PatientBasicUpdate,
    PatientProfileOut,
)
from src.db.postgres.models_patient import (
    Allergy,
    MedicalHistory,
    Medication,
    MenstrualReproductive,
    Patient,
    User,
)


_logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────────────────────────────────────
# 主档案
# ────────────────────────────────────────────────────────────────────────────


@router.get(
    "/me",
    response_model=PatientProfileOut,
    summary="读自己档案(主表 + 8 张子表汇总)",
)
def get_my_profile(
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> PatientProfileOut:
    user: User | None = db.get(User, current_user.user_id)
    if user is None:
        raise HTTPException(404, "用户不存在")

    patient: Patient | None = db.get(Patient, current_user.user_id)
    history = load_medical_history(current_user.user_id)  # 8 张子表汇总

    return PatientProfileOut(
        user_id=current_user.user_id,
        email=user.email,
        name=patient.name if patient else None,
        gender=patient.gender if patient else None,
        birth_date=patient.birth_date if patient else None,
        blood_type=patient.blood_type if patient else None,
        height_cm=patient.height_cm if patient else None,
        weight_kg=patient.weight_kg if patient else None,
        phone=patient.phone if patient else None,
        emergency_contact=patient.emergency_contact if patient else None,
        personal_history=history["personal_history"],
        past_history=history["past_history"],
        allergy_history=history["allergy_history"],
        medication_history=history["medication_history"],
        family_history=history["family_history"],
        obstetric_history=history["obstetric_history"],
    )


@router.put(
    "/me",
    response_model=PatientProfileOut,
    summary="改自己的基本信息 + 个人史(patients 表本身字段)",
)
def update_my_profile(
    payload: PatientBasicUpdate,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> PatientProfileOut:
    patient: Patient | None = db.get(Patient, current_user.user_id)
    if patient is None:
        # 首次更新 = 自动创建一行(reg 时只建 users,patients 行延后到首次填档案)
        patient = Patient(id=current_user.user_id)
        db.add(patient)

    # 只 update 显式给出的字段(exclude_unset 区分 None 和未传)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(patient, k, v)

    db.flush()
    # patient_repo.load_medical_history 走独立 session_scope,看不到本会话未 commit
    # 的更新 → 必须 commit 让其他会话可见。get_db 的 finally 会再 commit 一次,无副作用
    db.commit()
    db.refresh(patient)
    return get_my_profile(current_user=current_user, db=db)


# ────────────────────────────────────────────────────────────────────────────
# 三张 ⚠️必问表的 Create / Delete
# ────────────────────────────────────────────────────────────────────────────


def _ensure_patient_row(current_user: CurrentUser, db: OrmSession) -> None:
    """子表都 FK→patients.id ON DELETE CASCADE,但 patients 行需先存在。
    用户没填基本信息也要让他能添加病史/过敏 → 自动建空 patients 行。"""
    if db.get(Patient, current_user.user_id) is None:
        db.add(Patient(id=current_user.user_id))
        db.flush()


def _delete_owned_or_404(
    db: OrmSession, model_cls, record_id: str, current_user: CurrentUser
) -> None:
    """通用:按 id 拿子表行,owner 不符或不存在都 → 404(防泄漏存在性)。"""
    row = db.get(model_cls, record_id)
    if row is None or row.patient_id != current_user.user_id:
        raise HTTPException(404, "记录不存在或无权访问")
    db.delete(row)
    db.flush()


# ── medical_history ────────────────────────────────────────────────────


@router.post(
    "/me/medical-history",
    response_model=CreatedRecordOut,
    status_code=status.HTTP_201_CREATED,
)
def create_medical_history(
    payload: MedicalHistoryCreate,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> CreatedRecordOut:
    _ensure_patient_row(current_user, db)
    row = MedicalHistory(patient_id=current_user.user_id, **payload.model_dump())
    db.add(row)
    db.flush()
    db.refresh(row)
    return CreatedRecordOut(id=row.id)


@router.delete(
    "/me/medical-history/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_medical_history(
    record_id: str,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> None:
    _delete_owned_or_404(db, MedicalHistory, record_id, current_user)


# ── allergies ───────────────────────────────────────────────────────────


@router.post(
    "/me/allergies",
    response_model=CreatedRecordOut,
    status_code=status.HTTP_201_CREATED,
)
def create_allergy(
    payload: AllergyCreate,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> CreatedRecordOut:
    _ensure_patient_row(current_user, db)
    row = Allergy(patient_id=current_user.user_id, **payload.model_dump())
    db.add(row)
    db.flush()
    db.refresh(row)
    return CreatedRecordOut(id=row.id)


@router.delete(
    "/me/allergies/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_allergy(
    record_id: str,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> None:
    _delete_owned_or_404(db, Allergy, record_id, current_user)


# ── medications ─────────────────────────────────────────────────────────


@router.post(
    "/me/medications",
    response_model=CreatedRecordOut,
    status_code=status.HTTP_201_CREATED,
)
def create_medication(
    payload: MedicationCreate,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> CreatedRecordOut:
    _ensure_patient_row(current_user, db)
    row = Medication(patient_id=current_user.user_id, **payload.model_dump())
    db.add(row)
    db.flush()
    db.refresh(row)
    return CreatedRecordOut(id=row.id)


@router.delete(
    "/me/medications/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_medication(
    record_id: str,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> None:
    _delete_owned_or_404(db, Medication, record_id, current_user)


# ── obstetric (1:1) ─────────────────────────────────────────────────────


@router.put(
    "/me/obstetric",
    response_model=ObstetricOut,
    summary="婚育/月经史 upsert(女性 1:1 子表;⑪ safety_gate 妊娠/哺乳禁忌兜底硬依赖)",
)
def upsert_my_obstetric(
    payload: ObstetricUpdate,
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> ObstetricOut:
    """1:1 sub-table:行不存在则创建,存在则 partial update。

    `payload.model_dump(exclude_unset=True)` 区分"传 None 显式清除"和"未传保持原值"。
    """
    _ensure_patient_row(current_user, db)
    row: MenstrualReproductive | None = db.execute(
        select(MenstrualReproductive).where(
            MenstrualReproductive.patient_id == current_user.user_id
        )
    ).scalar_one_or_none()
    if row is None:
        row = MenstrualReproductive(patient_id=current_user.user_id)
        db.add(row)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(row, k, v)
    db.flush()
    db.refresh(row)
    return ObstetricOut(
        menarche_age=row.menarche_age,
        cycle_days=row.cycle_days,
        period_days=row.period_days,
        last_menstrual_period=row.last_menstrual_period,
        is_pregnant=row.is_pregnant,
        gravidity=row.gravidity,
        parity=row.parity,
        is_lactating=row.is_lactating,
        menopause_age=row.menopause_age,
        notes=row.notes,
    )


@router.delete(
    "/me/obstetric",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_my_obstetric(
    current_user: CurrentUser = Depends(require_role("patient")),
    db: OrmSession = Depends(get_db),
) -> None:
    """删除该患者整行 obstetric 记录;行不存在则幂等返 204。"""
    row: MenstrualReproductive | None = db.execute(
        select(MenstrualReproductive).where(
            MenstrualReproductive.patient_id == current_user.user_id
        )
    ).scalar_one_or_none()
    if row is None:
        return  # 幂等
    db.delete(row)
    db.flush()
