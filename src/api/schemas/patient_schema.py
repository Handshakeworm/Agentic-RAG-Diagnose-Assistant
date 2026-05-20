"""患者档案接口 Pydantic 模型(DEV_SPEC §8.4 G5 / §2.4.5)。

字段对齐 ORM(`models_patient.py`)。Pydantic 校验前置,DB 兜后备约束。

scope:
- `PatientProfileOut` 全档案聚合(8 张表 + 主表),`GET /patients/me` 用
- `PatientBasicUpdate` 主表 patients + 个人史字段更新(`PUT /patients/me`)
- `MedicalHistoryCreate` / `AllergyCreate` / `MedicationCreate` 三张 ⚠️必问表
  的创建(`POST /patients/me/<section>`)+ 对应 Out 模型回显

其余 5 张子表(surgical_trauma / transfusion / family_history / menstrual /
exam_reports)在 `PatientProfileOut` 里读出,但独立 CRUD 端点本期不实现 —
demo 场景下用户填好这些走 admin 后台或直接 SQL 即可。
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────────────
# 主档案
# ────────────────────────────────────────────────────────────────────────────


class PatientBasicUpdate(BaseModel):
    """PUT /patients/me 请求体 — 全部字段可选,只 update 显式给出的。"""

    name: str | None = None
    gender: Literal["male", "female", "other"] | None = None
    birth_date: date | None = None
    blood_type: str | None = Field(None, max_length=20)
    height_cm: int | None = Field(None, ge=30, le=300)
    weight_kg: Decimal | None = Field(None, ge=0, le=999.9)
    phone: str | None = None
    emergency_contact: str | None = None

    smoking_status: Literal["never", "former", "current"] | None = None
    smoking_pack_years: Decimal | None = Field(None, ge=0, le=999.9)
    alcohol_status: Literal["never", "occasional", "regular", "heavy"] | None = None
    alcohol_detail: str | None = None
    occupation: str | None = None
    occupational_exposure: str | None = None
    travel_history: str | None = None
    infectious_contact: str | None = None


class PatientProfileOut(BaseModel):
    """GET /patients/me 响应体 — 主表 + 8 张子表汇总。

    复用 `agent.utils.patient_repo.load_medical_history` 已经定义好的 dict 形状,
    避免在两处重复字段映射(load_medical_history 是 Agent 流水线的真实消费方,
    G5 暴露相同的 shape 让前端跟 Agent 看到的数据一致)。
    """

    user_id: str
    email: str

    name: str | None
    gender: str | None
    birth_date: date | None
    blood_type: str | None
    height_cm: int | None
    weight_kg: Decimal | None
    phone: str | None
    emergency_contact: str | None

    personal_history: dict[str, Any]
    past_history: dict[str, Any]
    allergy_history: list[dict[str, Any]]
    medication_history: list[dict[str, Any]]
    family_history: list[dict[str, Any]]
    obstetric_history: dict[str, Any] | None


# ────────────────────────────────────────────────────────────────────────────
# 三张 ⚠️必问表的 Create 请求
# ────────────────────────────────────────────────────────────────────────────


class MedicalHistoryCreate(BaseModel):
    """既往病史(基础疾病 + 传染病)。spec §2.4.5。"""

    category: Literal["chronic", "infectious"]
    condition: str = Field(..., min_length=1, max_length=200)
    icd10_code: str | None = Field(None, max_length=10)
    diagnosed_at: date | None = None
    resolved_at: date | None = None
    control_status: Literal["well_controlled", "poorly_controlled", "unknown"] | None = None
    notes: str | None = None


class AllergyCreate(BaseModel):
    """过敏史 ⚠️ 安全底线(safety_gate ⑪ 规则层直接读)。spec §2.4.5。"""

    allergen: str = Field(..., min_length=1, max_length=200)
    allergen_type: Literal["drug", "food", "environmental", "material", "other"] | None = None
    reaction: str | None = None
    reaction_type: str | None = Field(None, max_length=30)
    severity: Literal["mild", "moderate", "severe", "life_threatening"] | None = None
    status: Literal["confirmed", "suspected", "resolved"] = "suspected"


class MedicationCreate(BaseModel):
    """用药史(当前 + 历史)。`ended_at=None` 表示当前正在服用。spec §2.4.5。"""

    drug_name: str = Field(..., min_length=1, max_length=200)
    drug_category: str | None = Field(None, max_length=30)
    dosage: str | None = None
    frequency: str | None = None
    route: Literal["oral", "injection", "topical", "inhalation", "other"] | None = None
    started_at: date | None = None
    ended_at: date | None = None
    prescribed_by: str | None = None
    is_self_medication: bool = False


class CreatedRecordOut(BaseModel):
    """三张子表 POST 通用响应:返回新行的 id,前端拿来做后续 DELETE。"""

    id: str


# ────────────────────────────────────────────────────────────────────────────
# 婚育/月经史(menstrual_reproductive,1:1)— PUT upsert 语义
# ────────────────────────────────────────────────────────────────────────────


class ObstetricUpdate(BaseModel):
    """PUT /patients/me/obstetric 请求体 — 1:1 子表 upsert,全字段可选。

    设计依据:此表是 ⑪ safety_gate 规则层硬依赖(妊娠/哺乳禁忌兜底),
    必须有写入路径,否则女性患者用药安全永远兜不住(spec §4.1.2 ⑪)。
    """

    menarche_age: int | None = Field(None, ge=8, le=20)
    cycle_days: int | None = Field(None, ge=14, le=90)
    period_days: int | None = Field(None, ge=1, le=15)
    last_menstrual_period: date | None = None
    is_pregnant: bool | None = None
    gravidity: int | None = Field(None, ge=0, le=30)
    parity: int | None = Field(None, ge=0, le=30)
    is_lactating: bool | None = None
    menopause_age: int | None = Field(None, ge=20, le=80)
    notes: str | None = None


class ObstetricOut(BaseModel):
    """PUT /patients/me/obstetric 响应体 — 完整回显当前 obstetric 行。"""

    menarche_age: int | None
    cycle_days: int | None
    period_days: int | None
    last_menstrual_period: date | None
    is_pregnant: bool | None
    gravidity: int | None
    parity: int | None
    is_lactating: bool | None
    menopause_age: int | None
    notes: str | None
