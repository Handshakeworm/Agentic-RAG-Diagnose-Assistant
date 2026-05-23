"""Agent ① info_collect Step 1 LLM 输出 schema(DEV_SPEC §9.5 第 1 项)。

`InfoCollectOutput` 是 ① info_collect 节点 Step 1 的 LLM 结构化输出契约,
被 `src/agent/nodes/info_collect.py` 通过 `llm.with_structured_output()` 调用。

字段定义直接来源于 §9.5 第 1 项,**禁止私改**——spec authority hierarchy 规定
§9.5 是权威,业务代码必须对齐。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PresentIllnessSlots(BaseModel):
    """现病史结构化要素槽位(13 个维度),未提及的维度为 None/空列表。

    与 src/agent/state.py 的 PresentIllnessSlots 字段一一对应——同一份契约,
    schema 这边给 LLM 看(用于 with_structured_output),state 那边给运行时存。
    """

    onset_time:          str | None = Field(None, description="起病时间,如'3天前'")
    onset_mode:          str | None = Field(None, description="起病方式:急性/缓慢/隐匿")
    trigger:             list[str] = Field(default_factory=list, description="诱因(多值):劳累/受凉/进食/熬夜等可叠加")
    location:            str | None = Field(None, description="部位")
    nature:              list[str] = Field(default_factory=list, description="性质(多值):刺痛/胀痛/绞痛/烧灼感可同时存在")
    severity:            list[str] = Field(default_factory=list, description="程度(多值):主观描述+0-10 NRS 评分可叠加,如['影响睡眠','7-8 分']")
    duration_pattern:    str | None = Field(None, description="时间规律:持续性/间歇性/阵发性")
    aggravating:         list[str] = Field(default_factory=list, description="加重因素")
    relieving:           list[str] = Field(default_factory=list, description="缓解因素")
    associated_symptoms: list[str] = Field(default_factory=list, description="伴随症状(患者自述)")
    progression:         str | None = Field(None, description="病程演变:加重/减轻/稳定/波动")
    treatments:          list[str] = Field(
        default_factory=list,
        description=(
            "诊疗经过(多值,半结构化):每条记录格式 '<治疗>: <反应>',如 "
            "['布洛芬: 无效', '热敷: 部分缓解'];多次治疗各成一条,解决 tried/response 跨字段对齐"
        ),
    )


class InfoCollectOutput(BaseModel):
    """① info_collect LLM 输出。

    本节点单次 LLM 同时处理两路输入,因此输出含两组字段:
      - **patient_input 抽取**:`chief_complaint` / `present_illness` / `present_illness_slots`
      - **⓪a form 答案解析**:`history_fills` / `obstetric_fills` / `new_symptoms`
        (form 含 type=open 时填 new_symptoms;type=history 时填 history_fills;
         type=obstetric 时填 obstetric_fills;无对应 type 则保持默认空)

    合并是为了让"⓪a → ⑦ → ①"链路压成"⓪a → ①"一次 LLM 调用 —— ⑦ 在首轮 to_info_collect
    路径上的工作完全可以由 ① 一并做掉,职责更内聚。
    """

    # ─── patient_input 抽取(主诉 / HPI) ───
    chief_complaint:       str = Field(..., description="主诉(主要症状+持续时间),如'腹痛3天';来自 patient_input")
    present_illness:       str = Field(..., description="现病史自由文本(本次发病的详细展开);来自 patient_input")
    present_illness_slots: PresentIllnessSlots = Field(
        ..., description="现病史结构化槽位,与 present_illness 同步填充;主要来自 patient_input,form 答案附带的细节也可填入"
    )

    # ─── ⓪a form 答案解析(原 FollowupParseResult 同构,只是 ① 节点合并承担) ───
    # 症状三分类:明确确认 / 明确否认 / 模糊不定。LLM 按患者语气判,
    # 下游 ⑤ generate_followup 拿三个列表去重,避免重复追问已知症状。
    confirmed_symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "⓪a form 的 open 题 + patient_input 里**明确确认**的症状(语气肯定):"
            "如'右上腹疼'/'有点恶心'。**只含 form/自述里提到的,主诉本身不要重复**。"
        ),
    )
    denied_symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "⓪a form 的 open 题 + patient_input 里**明确否认**的症状(原文有'没'/'不'+症状):"
            "如'没吐'/'不发烧'/'没腹泻'。**只识别明确否认**,完全没提到的不要列。"
        ),
    )
    uncertain_symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "⓪a form 的 open 题 + patient_input 里**模糊/不确定**语气提及的症状:"
            "如'可能有点头晕'/'好像偶尔会咳'/'不太确定有没有发烧'。"
            "区别于 confirmed(语气肯定)和 denied(明确否认)。"
        ),
    )
    obstetric_fills: dict[str, bool | None] | None = Field(
        None,
        description=(
            "仅当 ⓪a form 含 type=obstetric 时填,key ∈ {is_pregnant, is_lactating},"
            "value=true(确认)/ false(明确否认)/ None(回答不明)。"
            "节点会写回 state.medical_history['obstetric_history'],供 ⑪ safety_gate 使用。"
        ),
    )
    history_fills: dict[str, list[str]] | None = Field(
        None,
        description=(
            "仅当 ⓪a form 含 type=history 时填(入站病史采集)。三段扁平 list[str]:"
            "  - allergies:过敏原名"
            "  - medications:在用/长期用药名"
            "  - past_conditions:既往疾病名"
            "节点按 patient_repo.load_medical_history 的字段映射追加到 medical_history 对应表。"
            "患者回答'无'/'没有'→ 三段空 list(显式区分'已问无答' vs '未问')。"
        ),
    )
