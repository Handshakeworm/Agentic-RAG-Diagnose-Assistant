"""Agent ⑦ process_followup_answer LLM 输出 schema(DEV_SPEC §9.5 第 7 项)。

⑦ 解析患者对 ⑤ 追问的回答,产出五类信息:
  - slot_fills:         维度级回填(对应 type=slot 追问项)
  - confirmed_symptoms: 患者明确确认的症状(语气肯定)→ state.confirmed_symptoms
  - denied_symptoms:    患者明确否认的症状(原文带'没'/'不')→ state.denied_symptoms
  - uncertain_symptoms: 患者模糊语气提及的症状(犹豫/可能)→ state.uncertain_symptoms
  - obstetric_fills:    妊娠/哺乳状态(对应 type=obstetric 追问项)
  - history_fills:      既往史/过敏史/用药史(对应 type=history 追问项,⓪a 入站追问产物)

症状三分类:LLM 按患者语气把症状归到 confirmed/denied/uncertain 任一类,intake 不经过
② build_query NER negation 分流,这里不写 denied/uncertain 的话 ⑤ 永远看不到。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FollowupParseResult(BaseModel):
    """⑦ process_followup_answer LLM 输出。"""

    slot_fills: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description=(
            "维度级回填,key=槽位名;value 类型与 PresentIllnessSlots 槽位一致:"
            "单值槽(onset_time/onset_mode/location/duration_pattern/progression)为 str,"
            "多值槽(trigger/nature/severity/aggravating/relieving/associated_symptoms/treatments)为 list[str]"
            "(treatments 每条半结构化 '<治疗>: <反应>')"
        ),
    )
    confirmed_symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "患者**明确确认**的症状(语气肯定、不带犹豫):"
            "如'右上腹疼'/'有点恶心'/'每天都拉肚子'。"
            "用患者原文或常见医学短语,不要太长。下游 ⑤ 用此列表去重。"
        ),
    )
    denied_symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "患者**明确否认**的症状(原文出现'没'/'不'/'没有'+症状名 的组合):"
            "如'有点恶心,没吐'→ denied_symptoms=['呕吐'];'不痛'→ ['疼痛'];"
            "'没腹泻'→ ['腹泻']。下游 ⑤ 用此列表去重避免重复问。"
            "**只识别明确否认**(伴随明显否定词);患者完全没提到的症状不要列。"
        ),
    )
    uncertain_symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "患者**模糊/不确定**语气提及的症状(带犹豫词):"
            "如'可能有点头晕'/'好像偶尔会咳'/'不太确定有没有发烧'→ uncertain。"
            "区别于 confirmed(语气肯定)和 denied(明确否认)。"
            "下游 ⑤ 看到 uncertain 列表里有 X,可以追问 X 详情但不当作铁证。"
        ),
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
            "病史回填,**5 段** list[str](2026-05-22 从 3 段扩到 5 段,覆盖家族史/个人史)。"
            "本轮含 type=history(⓪a 入站)或 ⑤ Step A targeted 题命中史类语义时启用,"
            "LLM 按 question 内容自动判断要填哪几段:"
            "  - allergies:过敏原名(青霉素 / 海鲜 ...)"
            "  - medications:在用/长期用药名(氯沙坦 / 二甲双胍 ...)"
            "  - past_conditions:既往**疾病**名(高血压 / 糖尿病 / 胆囊炎史 ...) — 疾病史非症状史"
            "  - family_conditions:家族史,半结构化:"
            "      '<关系>:<疾病>[(发病<年龄>岁)]' 表达阳性(如 '父亲:糖尿病(发病50岁)');"
            "      '<疾病>:无' 表达已问无(如 '胆结石:无');"
            "      ⑦ 解析后:阳性 entry 入 medical_history.family_history,"
            "      已问无入 medical_history.family_history_asked_no"
            "  - personal_conditions:个人史(烟酒/职业/旅居/暴露),半结构化:"
            "      '<项目>:<详细>' 表达内容(如 '吸烟:每天1包15年');"
            "      '<项目>:无' 表达已问无(如 '饮酒:无');"
            "      ⑦ 解析后:有内容入 medical_history.personal_history.dynamic_notes,"
            "      已问无入 medical_history.personal_history_asked_no"
            "整段缺省(不写) = 本轮无史类问询;空 list = 本轮问了但 LLM 没识别到任何信息。"
        ),
    )
