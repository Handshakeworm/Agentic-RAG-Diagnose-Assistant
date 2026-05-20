"""Agent ⑦ process_followup_answer LLM 输出 schema(DEV_SPEC §9.5 第 7 项)。

⑦ 解析患者对 ⑤ 追问的回答,产出四类信息:
  - slot_fills:       维度级回填(对应 type=slot 追问项)
  - new_symptoms:     患者回答中主动提到的新症状(对应 type=open 追问项,或顺带补充)
  - obstetric_fills:  妊娠/哺乳状态(对应 type=obstetric 追问项)
  - history_fills:    既往史/过敏史/用药史(对应 type=history 追问项,⓪a 入站追问产物)
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FollowupParseResult(BaseModel):
    """⑦ process_followup_answer LLM 输出。"""

    slot_fills: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description=(
            "维度级回填,key=槽位名;value 类型与 PresentIllnessSlots 槽位一致:"
            "单值槽(onset_time/onset_mode/trigger/location/nature/severity/"
            "duration_pattern/progression/treatment_tried/treatment_response)为 str,"
            "多值槽(aggravating/relieving/associated_symptoms)为 list[str]"
        ),
    )
    new_symptoms: list[str] = Field(
        default_factory=list,
        description="患者回答中提及的新症状(开放式追问的主要产物,也包括患者顺带补充)",
    )
    obstetric_fills: dict[str, bool | None] | None = Field(
        None,
        description=(
            "仅当本轮含 type=obstetric 追问时填,key ∈ {is_pregnant, is_lactating},"
            "value=true(确认)/ false(明确否认)/ None(回答不明)。"
            "⑦ 会把这两字段写回 state.medical_history['obstetric_history'],"
            "供 ⑪ safety_gate 妊娠/哺乳禁忌兜底使用。"
        ),
    )
    history_fills: dict[str, list[str]] | None = Field(
        None,
        description=(
            "仅当本轮含 type=history 追问时填(⓪a 入站病史采集)。"
            "三段扁平 list[str]:"
            "  - allergies:过敏原名(青霉素 / 海鲜 ...)"
            "  - medications:在用/长期用药名(氯沙坦 / 二甲双胍 ...)"
            "  - past_conditions:既往疾病名(高血压 / 糖尿病 ...)"
            "⑦ 会按 patient_repo.load_medical_history 的字段映射追加到 "
            "medical_history['allergy_history'] / ['medication_history'] / "
            "['past_history']['medical_history'],去重后供 ⑪ safety_gate 使用。"
            "患者回答'无'/'没有'→ 三段空 list(显式区分'已问无答' vs '未问')。"
        ),
    )
