"""src/prompts/agent.py — Agent 各 LLM 调用点的 prompt 构造函数(DEV_SPEC §4.1.2)。

每个函数返回**一段文本字符串**(消息内容),由调用方包成 LangChain 消息后
喂给 `chain.invoke(prompt, config={...})`。

设计约定:
- 一个调用点一个函数,函数名与 §9.3 调用点名对应
- 函数签名只接 plain Python 数据(str / list / dict),不依赖 MedicalState
  类型 —— 调用方从 state 抽字段后传入,prompt 模块不感知 state 对象
- prompt 内联 schema 字段说明,降低对 §9.5 的查阅依赖;LLM 自己从
  `with_structured_output` 拿严格 schema,prompt 文本作为补充语义提示
- 多模态调用(① .5 / ⑨ / ⑩ Step 1)返回 `(messages, prompt_text)` 二元组:
  `messages: list[BaseMessage]` 供 chain.invoke 消费(含图);
  `prompt_text: str` 供 §9.6 `final_prompt` 审计存档(纯文本镜像)

每个 prompt 都尽量短小聚焦:同一节点不同 step 的 prompt 各自独立,避免一个
"上帝 prompt"什么都管。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage


# ────────────────────────────────────────────────────────────────────────────
# JSON 输出尾巴 — 仅给 method="json_mode" 的主链 12 处 prompt 拼接
# ────────────────────────────────────────────────────────────────────────────
# DeepSeek 主链不支持 function_calling / json_schema,只能用 method="json_mode";
# 走 json_mode 时 OpenAI 兼容协议要求 prompt 含 "json" 字样(否则 400)。
# enrichment.py 的 prompt 自带 4-field JSON 输出指引(SHARED_4FIELD_TAIL)是同样
# 的逻辑;agent prompt 没自带,所以统一拼这条尾巴。
#
# vision LLM(qwen via DashScope)走默认 function_calling(避开 thinking + json_mode
# 冲突,见 scripts/figure_enrichment_generation.py 注释),不需要这条尾巴 ——
# 所以 build_evidence_assembly_prompt / build_report_parsing_prompt 两个 vision
# 调用的 prompt **不**拼接此尾巴。
_JSON_TAIL = "\n\n请严格按 JSON 格式输出,所有字段值与上述 schema 描述一致。"


def _format_slot_value(v) -> str:
    """slot 值渲染到 prompt 文本:list[str] 用顿号串,str 直接返回,None/空返空串。

    2026-05-22 多值槽扩到 6 个后,prompt 拼 `f"{k}: {v}"` 直接显示 Python repr
    (`['进食', '熬夜']`),不漂亮也降低 LLM 可读性。统一在这里转成自然语言。
    """
    if isinstance(v, list):
        return "、".join(str(x) for x in v if x)
    return str(v) if v is not None else ""


# ────────────────────────────────────────────────────────────────────────────
# ① info_collect Step 1
# ────────────────────────────────────────────────────────────────────────────


def build_info_collect_prompt(
    patient_input: str,
    initial_form_question: str = "",
    initial_form_answer: str = "",
) -> str:
    """① info_collect:一次 LLM 同时处理 patient_input + ⓪a 综合 form 答案。

    输出由 `InfoCollectOutput` schema 严格约束,字段分两组:
      - 从 patient_input 抽:chief_complaint / present_illness / present_illness_slots
      - 从 ⓪a form 答案抽:history_fills / obstetric_fills / new_symptoms
    """
    has_form = bool((initial_form_question or "").strip()) and bool(
        (initial_form_answer or "").strip()
    )
    form_block = (
        f"""

【⓪a 综合 form 问答(open + history + obstetric,若女性)】
问:
{initial_form_question}

答:
{initial_form_answer}
"""
        if has_form
        else ""
    )
    form_extraction_lines = (
        """
4. 症状三分类(从 ⓪a form open 题答案 + patient_input 自由文本里识别;**主诉本身的症状不要重复**):
   - **confirmed_symptoms**(语气肯定):"右上腹疼"/"有点恶心"
   - **denied_symptoms**(明确否认,原文有'没'/'不'+症状):"没吐"/"不发烧"
   - **uncertain_symptoms**(模糊/犹豫):"可能有点头晕"/"好像偶尔会咳"
   - 同一句"有 A 没 B" → confirmed=[A], denied=[B];"可能 C" → uncertain=[C]
   - 患者**完全没提到**的症状任何一类都不要列(只识别明确说了的)
   - **patient_input 自由文本里 present_illness_slots 已捕获的字段**(如 location、nature 描述)
     **里如果含独立症状/体征属性**(放射、伴随表现、性质本身、部位扩展等),除了填进 slots,
     **也要按语气归到三类**:
     - 例 patient_input 含"右上腹疼,有时往后背窜" → slots.location 照写 + confirmed_symptoms 加"右上腹疼痛"、"放射至背部"
     - **跳过 `associated_symptoms` slot 里的内容**(它本身就是症状清单,不要重复写到 confirmed_symptoms)
5. history_fills(仅 form 含 history 题时):
   - allergies: 过敏原名,如 ["青霉素", "海鲜"]
   - medications: 在用/长期用药名,如 ["氯沙坦"]
   - past_conditions: 既往疾病名,如 ["高血压", "糖尿病"]
   - 患者明确答"无 / 没有 / 都没有"→ 三段都填 []  (显式区分"已问无答" vs "未问")
   - 患者没答这部分 → 字段保持 null
6. obstetric_fills(仅 form 含 obstetric 题时,且患者为女性):
   - is_pregnant: true / false / null(回答不明)
   - is_lactating: true / false / null
   - 患者没答这部分 → 字段保持 null
"""
        if has_form
        else ""
    )

    return f"""你是医院分诊台问诊助手。下面是患者的自述。请仅从该自述中提取本次就诊的信息,
**不要自行编造、不要泛化、不要补充未提及的内容**。

【患者自述】
{patient_input}{form_block}
【提取要求】
1. chief_complaint(主诉):主要症状 + 持续时间,1 句话,例:"腹痛3天" —— **来源限 patient_input**
2. present_illness(现病史):用 1-3 句话展开本次发病的:起病时间、诱因、症状特点
   (部位/性质/程度)、伴随症状、加重/缓解因素、治疗经过 —— **主要来源 patient_input**
3. present_illness_slots(12 维结构化槽位)—— **主要来源 patient_input**,form 答案附带的细节
   也可填入:
   - 单值槽(str | None):onset_time / onset_mode / location / duration_pattern / progression
   - 多值槽(list[str]):aggravating(加重因素) / relieving(缓解因素) /
     associated_symptoms(伴随症状) / trigger(诱因,可叠加) / nature(性质,可多) /
     severity(程度,主观描述 + NRS 评分可叠加) /
     treatments(诊疗经过,每条半结构化 '<治疗>: <反应>',如 ['布洛芬: 无效', '热敷: 部分缓解'])
   - 患者**未提及**的维度严格保持 None / 空列表,**不要瞎填**{form_extraction_lines}

注意:这是初诊采集,信息缺失是正常的,后续会通过追问补全。""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ①.5 analyze_initial_reports / ⑨ process_exam_result
# ────────────────────────────────────────────────────────────────────────────


def build_report_parsing_prompt(
    num_reports: int,
    hint: str | None = None,
) -> str:
    """①.5 / ⑨ 多模态 LLM 直读报告 → 结构化关键发现。

    Args:
        num_reports: 本次解析的报告数量,prompt 中提示 LLM 输出对应数量的 finding 项
        hint: 可选上下文提示,如"这组报告是 <group_label>,期望含 <items>"。
              ⑨ 按 group 分别调用时传入,帮 LLM 定位报告类型(化验 vs 影像 vs 病历)
              提升解析准确度。①.5 入站解析(患者一次混传)时不带 hint。

    多模态附件(图片 base64 / PDF 文件)由调用方组装到 LangChain message 中,
    本函数只产文本提示部分。
    """
    hint_block = ""
    if hint:
        hint_block = f"\n【上下文提示】\n{hint}\n"

    return f"""你是医学报告结构化解析助手。下面附了 {num_reports} 份检查报告(图片或 PDF)。
请逐份解析,产出结构化的 ReportFindings 列表,每份报告对应 findings 列表中的一项。
{hint_block}
【提取规则】
- report_type:从 ['blood_routine','urine_routine','biochemistry','imaging','ecg',
  'physical_exam','pathology','other'] 中选最贴切的
- report_date:从报告头/落款抽取日期,YYYY-MM-DD 格式;识别不到 → null
- abnormal_values:**保留原始数值**,如 "WBC 12.3×10⁹/L↑" "Hb 85g/L↓",不要意译为
  "白细胞高"
- impressions:报告诊断印象原文,如 "右肺上叶磨玻璃结节"
- positive_findings:阳性发现 + **异常值的临床解读**(如 WBC↑ → "白细胞升高"、
  Hb↓ → "贫血"),用医学文献语言,可直接用于 query 召回
- negative_findings:阴性发现 / 已排除项,如 "未见肝内胆管扩张"、"肝功能正常"

报告本身已是标准医学术语,**不需要做实体链接**,直接读图/读字面提取。
若某类发现报告中不存在,对应字段返回空列表,不要编造。"""


# ────────────────────────────────────────────────────────────────────────────
# ② build_query Step 1 / 2 / 4
# ────────────────────────────────────────────────────────────────────────────


def build_ner_prompt(text: str) -> str:
    """② build_query Step 1:LLM NER 从文本抽取医学实体。"""
    return f"""你是医学命名实体识别助手。请从下面的患者陈述中抽取**医学实体**。

【输入文本】
{text}

【实体类型】(entity_type 取值)
- symptom(症状):如"头痛"、"恶心"、"胸闷"
- disease(疾病):如"糖尿病"、"高血压"
- drug(药物):如"二甲双胍"、"奥美拉唑"
- anatomy(解剖部位):如"右上腹"、"胸骨后"

【字段说明】
- text:实体原文(保留患者口语,不归一)
- entity_type:实体类型(见上)
- certainty:确定性三态(按患者**语气**判,不要把"模糊"误标成"否认"):
  - "confirmed":语气肯定提及,如"头痛"/"右上腹疼"
  - "denied":明确否认,原文带"没"/"不"+症状,如"没头痛"/"不发烧"
  - "uncertain":模糊/犹豫语气,如"可能头痛"/"好像有点痛"/"不太确定"
- temporality:时间属性。current(本次/当前) / past(既往) / family(家族)
- value:量化值,如体温 "38.5°C"、持续时间 "3天";无则 null

不要重复抽取同一实体的不同表述(如同时抽"肚子疼"和"腹痛"),保留患者原始表述即可——
后续 Step 2 Entity Linking 会做术语标准化。""" + _JSON_TAIL


def build_query_construction_prompt(
    confirmed_symptoms: list[str],
    medical_history_summary: str,
    report_positive: list[str],
    report_impressions: list[str],
    filled_slots: dict[str, Any],
) -> str:
    """② build_query Step 3:LLM 改写 dense_query(单字段输出)。

    Sparse 路词袋由 Step 2 state 多字段直采(chief + slots + report findings)确定性
    产出,完全不进 LLM 视野;LLM 只负责整合证据成一句语义连贯的 dense 查询。
    """
    slots_lines = [f"  - {k}: {_format_slot_value(v)}" for k, v in filled_slots.items() if v]
    slots_block = "\n".join(slots_lines) if slots_lines else "  (无)"

    pos_block = "; ".join(report_positive) or "(无)"
    imp_block = "; ".join(report_impressions) or "(无)"
    sym_block = "、".join(confirmed_symptoms) or "(无)"

    return f"""你是医学检索 query 改写助手。请把患者已确认的证据整合成**一句语义连贯的自然语言查询**,
便于 Dense 向量检索召回相关医学文献 chunk。

【已确认症状】
{sym_block}

【现病史已填维度】
{slots_block}

【报告阳性发现】
{pos_block}

【报告诊断印象】
{imp_block}

【病史关键摘要】
{medical_history_summary or "(无)"}

【Dense Query 要求】
- 用医学文献风格,而不是患者口语
- 长度 ≤ 200 字,**完整保留所有客观信号**(已确认症状 + 已填 slots 维度 + 报告阳性/印象 + 病史关键摘要全要覆盖),不要为了精简而省略
- 不要否定词("没有发烧"不进 query)
- 原始数值不进 query(白细胞具体数字不写),但"白细胞升高"这种语义化描述可写
- 把鉴别特征写出来(如"进食后加重的上腹胀痛伴反酸,白细胞升高,既往糖尿病史")
- 输出仅一个字段:dense_query""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ④ select_discriminative_symptom — Smart followup(1 LLM)
# ────────────────────────────────────────────────────────────────────────────


def build_smart_followup_prompt(
    chief_complaint: str,
    present_illness: str,
    filled_slots: dict,
    empty_slots: list[str],
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    uncertain_symptoms: list[str],
    quota: int,
) -> str:
    """④ 1 次 LLM 同时出 questions(追问) + unaskable_symptoms(粗筛)。"""
    filled_lines = [f"  - {k}: {_format_slot_value(v)}" for k, v in filled_slots.items() if v]
    filled_block = "\n".join(filled_lines) if filled_lines else "  (无,全部空缺)"
    empty_block = ", ".join(empty_slots) or "(无,12 维已全部填满)"
    conf_block = "、".join(confirmed_symptoms) or "(无)"
    den_block = "、".join(denied_symptoms) or "(无)"
    unc_block = "、".join(uncertain_symptoms) or "(无)"

    return f"""你是临床问诊助手。请基于患者已提供的信息,同时产出本轮**追问项** + **想知道但患者答不上的体征**。

【患者主诉】{chief_complaint or "(无)"}
【现病史描述】{present_illness or "(无)"}

【已填的 HPI 维度】
{filled_block}
(若某维度 value 为 "(患者未明确)" 或列表含此值,表示**已问过但患者答不上**;
 视为该维度信息已采集完毕,**不要重复追问**;诊断推理时把它当作"患者明确否认/不知"看待)

【空缺的 HPI 维度】(12 维框架内尚未问到的)
{empty_block}

【已确认有的症状】{conf_block}
【已否认的症状】{den_block}
【已问但患者不确定的】{unc_block}

【任务 1:questions — 本轮追问项,0-{quota} 条】
从下面两种 type 里选,**最多 {quota} 条,可以 0 条**(信息已足时直接返空,流程跳诊断):

1. **type="slot"** — 补全 HPI 空缺维度
   - 优先从【空缺维度】里挑对当前主诉**诊断价值最高**的(通常是 trigger/location/
     nature/duration_pattern/aggravating/relieving 这类患者能直接答的维度)
   - 把 slot 名(如 "trigger" / "location")写到 `slot` 字段
   - 不要选已填的;不要重复

2. **type="open"** — 开放式问"还有别的不舒服吗?"
   - 适合用在:12 维已大部分填满 / 空缺维度都不重要 / 想兜底捕获遗漏症状
   - 一轮**最多 1 条** open(再多无意义,患者也想不出更多)
   - `slot` 字段留 None

【任务 2:unaskable_symptoms — 想知道但患者答不上的体征/指标,0-{quota} 条】
**最多 {quota} 条,可以 0 条**。每条带:
- `description`:医生侧语言,写"想查什么 / 想知道什么体征",如"腹部 B 超提示有无胆囊壁
  增厚"、"血常规白细胞与中性粒细胞分类"
- `reason`:为什么对鉴别诊断重要,如"关键鉴别胆囊炎 vs 胃炎"

这些是患者**无法靠口述回答**但医生靠经验知道"应该查一下"的项。**不要**把可问的症状写进来
(那应该走 questions / open),也**不要**直接写检查名(那是 ⑧ recommend_exam 的事)。

【判断原则】
- 两个任务**互斥**:已确认/否认/不确定的症状不要重问(open 里也不问);可问的维度走 questions,
  不要塞 unaskable
- 信息已足时,questions 和 unaskable 都可以返空,让流程尽快进诊断 — 不要为问而问
- 总数:questions ≤ {quota},unaskable ≤ {quota}(独立各自计数)""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ⑤ generate_followup
# (旧 build_followup_question_prompt 已删 —— ⑤ 改单职责 + ④/⑤/intake 都自带
# question_templates 模板, 不再需要 LLM 拼自然语言追问文案。详见 generate_followup.py)
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# ⑦ process_followup_answer
# ────────────────────────────────────────────────────────────────────────────


def build_followup_parse_prompt(
    followup_question: str,
    followup_answer: str,
    questions: list[dict],
) -> str:
    """⑦ 解析患者回答 → 维度槽位回填 + 新症状提取 + 妊娠/哺乳 + 病史 history_fills。"""
    items_lines = []
    has_obstetric = False
    has_history = False
    for q in questions:
        if q.get("type") == "slot":
            items_lines.append(f"  - 补全 HPI 维度 {q['slot']}(回填到 slot_fills)")
        elif q.get("type") == "open":
            items_lines.append("  - 开放式问『还有别的不舒服』(新症状回填到 new_symptoms)")
        elif q.get("type") == "obstetric":
            has_obstetric = True
            items_lines.append("  - 妊娠/哺乳状态(回填到 obstetric_fills)")
        elif q.get("type") == "history":
            has_history = True
            items_lines.append("  - 入站病史采集:过敏/慢病/长期用药(回填到 history_fills)")
        elif q.get("type") == "targeted":
            items_lines.append(
                "  - 针对性追问:新症状回填到 new_symptoms;若答案涉及具体 HPI 维度,顺手回填 slot_fills"
            )
    items_block = "\n".join(items_lines) if items_lines else "  (无)"

    obstetric_rule = ""
    if has_obstetric:
        obstetric_rule = """
3. obstetric_fills(本轮含妊娠/哺乳追问时必填):
   - key=is_pregnant:患者明确说"怀孕了"/"是的"→ true;明确说"没怀孕"/"不是"→ false;
     回答含糊("不知道"/"没测"/未提及)→ null
   - key=is_lactating:患者明确说"在哺乳"/"在喂奶"→ true;"不哺乳"/"没在喂"→ false;
     未提及 → null
   - **两个 key 都要出现**(即使 value 是 null),让下游知道"已问过"vs"没问过"
   - 不含妊娠/哺乳追问的本轮 → obstetric_fills 字段缺省(不写)"""

    # history_rule:has_history(⓪a form 含 type=history 题)必启用;否则按需启用
    # —— LLM 看 question 内容,涉及史类(比如家族史/平时烟酒/职业暴露/既往疾病/过敏/长期用药)
    # 就填对应段。不涉及就整个 history_fills 缺省。
    history_rule = """
4. history_fills(**5 段** list[str],按 question 内容自动判断要填哪几段):

   **何时填**:question 涉及"史"类信息,**比如**:
   - "您以前有过 X 病吗?" / "曾经诊断过哪些慢性病?" → past_conditions(疾病史)
   - "对什么过敏?" / "有药物过敏吗?" → allergies
   - "平时在吃什么药?" / "长期服药情况?" → medications
   - "您父母兄弟姐妹有人得过 X 吗?" / "家里有 X 病史吗?" → family_conditions
   - "您平时喝酒/吸烟吗?" / "职业有没有粉尘暴露?" / "去过疫区吗?" → personal_conditions
   - 问句**不涉及任何史**(如"现在哪里疼?")→ history_fills 整段缺省(不写)

   **5 段格式**:
   - allergies:过敏原名 list,如 ["青霉素", "海鲜"];问了无 → []
   - medications:在用/长期药名 list,如 ["氯沙坦"];问了无 → []
   - past_conditions:既往**疾病**名 list,如 ["高血压", "胆囊炎"];问了无 → []
   - family_conditions(2026-05-22 新):半结构化 list[str]:
       * 阳性:`"<关系>:<疾病>[(发病<年龄>岁)]"`,如 ["父亲:糖尿病(发病50岁)", "母亲:胆结石"]
       * 已问无:`"<疾病>:无"`,如 ["胆结石:无", "心脏病:无"]
       * 问了但全无 → 用 ["<问到的疾病>:无"] 表达,不要写 [](LLM 必须告诉下游问过什么)
   - personal_conditions(2026-05-22 新):半结构化 list[str]:
       * 有内容:`"<项目>:<详细>"`,如 ["吸烟:每天1包15年", "饮酒:偶尔啤酒", "职业:化工厂粉尘"]
       * 已问无:`"<项目>:无"`,如 ["饮酒:无", "吸烟:无"]
       * 问了但全无 → 同上,用 "<项目>:无" 表达

   **重要 - 史 vs 近期事件 双填判断**:
   一句话同时问"史"和"近期"时,**两段都要填**,不要二选一:
   - 例 Q="您平时喝酒吗?最近有没有饮酒?" + A="无"
     → history_fills.personal_conditions = ["饮酒:无"](平时 = 史)
     → denied_symptoms 加 "近期饮酒"(最近 = 近期事件)
   - 判断公式:
     * 含"平时/以前/家族/父母/兄弟姐妹/曾经" → 史 → history_fills 对应段
     * 含"最近/这次/现在/本次/这两天" → 近期 → confirmed/denied/uncertain
     * 两者并列 → **双填,分别写两段**

   **症状史 vs 疾病史**(常错):
   - "以前有过类似 X 痛?"(X 是症状名)→ 这是症状的**反复发作史**
     → denied_symptoms 加 "既往类似X痛",**不进 past_conditions**
   - "以前得过 X 病?"(X 是疾病名)→ 这是**疾病史**
     → past_conditions 加 "X",不进 denied_symptoms"""

    return f"""你是问诊回答解析助手。请把患者回答结构化。

【追问问题】
{followup_question}

【本轮追问的待回答项】
{items_block}

【患者回答】
{followup_answer}

【解析规则】
1. 维度级回填(slot_fills,key=槽位名):
   - 单值槽(onset_time/onset_mode/location/duration_pattern/progression):value=str
   - 多值槽(trigger/nature/severity/aggravating/relieving/associated_symptoms/treatments):value=list[str]
   - 槽位名必须是 HPI 12 维之一,不要新造槽名
   - **severity 槽特殊规则**:只填**主观严重度描述**(轻/中/重 / 影响睡眠/吃饭/活动 /
     0-10 NRS 评分)。**绝对不要填客观生命体征数值或化验值**(温度、血压、脉搏、SpO2、
     血糖、WBC 等)— 这些数值应写到 associated_symptoms 多值槽里。
       - ✅ severity=["影响睡眠", "7-8 分"] / severity=["重度"]
       - ❌ severity=["38℃"]   → 应是 associated_symptoms 加 "发热 38℃"
       - ❌ severity=["150/95"] → 应是 associated_symptoms 加 "BP 升高 150/95"
       - ❌ severity=["WBC 12.5"] → 化验值不进 slot,等 ⑨ 解析报告
   - **treatments 槽特殊规则**:每条记录半结构化 `"<治疗>: <反应>"`,治疗 + 反应
     合写一条,**不要拆成两个字段**(旧设计 treatment_tried/treatment_response 已合并)。
     患者答多种治疗时一样一条:
       - ✅ treatments=["布洛芬: 无效", "热敷: 部分缓解"]
       - ✅ treatments=["奥美拉唑: 显著好转"]
       - ❌ treatments=["布洛芬, 热敷", "无效, 部分缓解"](拆成 2 条平行)
       - 患者只说用了什么没说反应 → 反应写"未提及",如 "布洛芬: 未提及"
       - 患者只说"有效/无效"没说具体药 → 跳过(无法配对)
   - **本轮被询问到的 slot,患者明确答"不知道/不清楚/没注意/没有/无"等**否定或不知:
     在 slot_fills 中写入哨兵值标记"已问无答",避免下游循环重问:
       - 单值槽 value="(患者未明确)"
       - 多值槽 value=["(患者未明确)"]
     (不能省略不写,否则 intake 会反复重问;不能写空字符串,会被当成未填)
   - 患者**完全没涉及**的槽位(没被问也没主动说)**不要**出现在 slot_fills 里
2. 症状三分类(按患者**语气**判,LLM 自己决定每个症状归哪一类):
   - **confirmed_symptoms**(语气肯定):"右上腹疼"/"有点恶心"/"每天都拉肚子"
   - **denied_symptoms**(明确否认):
     a. 原文有'没'/'不'+症状:"没吐"/"不痛"/"没腹泻"
     b. **回答只有'无'/'没有'/'没'/'第一次'/'从未'/'否'等单纯否认词时**,
        **必看【追问问题】对应那一条**,从 question 里抽出靶点症状写到 denied:
        - 例 Q="您有没有便血?" + A="没有" → denied 加 "便血"
        - 例 Q="您以前有过类似的右上腹疼痛吗?" + A="第一次" → denied 加 "既往类似腹痛"
        - 例 Q="您有没有打寒战?" + A="没有" → denied 加 "寒战"
        - 例 Q="皮肤或眼睛发黄?" + A="无" → denied 加 "皮肤黄染" 和 "巩膜黄染"
        - 关键:answer 没症状名 → 从 question 提靶点,**不要因为 answer 文本里没症状就漏识别**
     b2. **串联问句多否认**(question 里用'和/或/、/逗号'串联多个鉴别点)—
         **每个点都要单独写一项**,不要合并成一项,也不要只挑一个写:
        - 例 Q="您发烧最高多少度?有没有打寒战?" + A="38度,没寒战"
          → confirmed 加 "发热 38℃",denied 加 "寒战"
        - 例 Q="便秘、腹泻或大便颜色变浅?" + A="无"
          → denied 加 "便秘" + "腹泻" + "大便颜色变浅"
        - 例 Q="尿频、尿急、尿痛或小便颜色异常?" + A="无"
          → denied 加 "尿频" + "尿急" + "尿痛" + "小便颜色异常"
     b3. **denied/confirmed 写入要带时态/维度后缀**,避免下轮看不出差异:
        - "发热 38℃" 而不是只写"发热"(带量化)
        - "右肩放射" / "背部放射"(各部位独立写)
        - "近期饮酒" 而不是 "饮酒"(史 vs 近期独立)
        - "既往胆囊炎" 而不是 "胆囊炎"(疾病史前缀,跟现症区分)
     c. **特别例外:type=open 的"还有没有别的不舒服"题**(对应 items_block 里
        "开放式问『还有别的不舒服』"那一条),答"无/没了/没有/没什么"等**只代表
        无新症状补充**,**不要**写到 denied(像 "其他不适"/"其他症状" 这种泛化词
        不是临床症状名,写进 denied 会污染下游决策)
     d. **史类范畴的否认不进 denied_symptoms,进 history_fills 对应段**:
        - 例 Q="您直系亲属有人得过胆结石?" + A="无"
          → history_fills.family_conditions = ["胆结石:无"],**不写 denied_symptoms**
        - 例 Q="您平时喝酒吗?" + A="不喝"
          → history_fills.personal_conditions = ["饮酒:无"],**不写 denied_symptoms**
        - 例 Q="您以前有过胆囊炎吗?" + A="没有"
          → history_fills.past_conditions = [],**不写 denied_symptoms**(疾病史归 past_conditions)
        - 但症状反复发作 ≠ 疾病史:Q="以前有过类似腹痛吗?" + A="无"
          → denied 加 "既往类似腹痛"(症状史归 denied,见 b 例 2)
   - **uncertain_symptoms**(模糊/犹豫):"可能有点头晕"/"好像偶尔会咳"/"不太确定有没有发烧"
   - 同一句"有 A 没 B" → confirmed=[A], denied=[B];"可能 C" → uncertain=[C]
   - 用患者原文或常见医学短语,不要太长(如"反酸"、"右上腹放射痛")
   - 已在【上下文已知】confirmed/denied/uncertain 任一列表里的术语**不要**重复输出
   - 患者**完全没提到**的症状**任何一类都不要列**(只识别患者明确说了的)
   - **抽取来源**:不光看 open / targeted 题答案;**slot 题答案里如果含独立症状/体征属性**
     (放射、伴随表现、性质本身、部位扩展等),除了填进 slot_fills,**也要按语气归到三类**:
     - 例 location 答"右上腹,有时往后背窜" → slot_fills.location 照写 + confirmed_symptoms 加"右上腹疼痛"、"放射至背部"
     - 例 nature 答"钝痛+胀,有时绞痛" → slot_fills.nature=["钝痛","胀痛","绞痛"](多值 list) + confirmed_symptoms 加"钝痛"、"胀痛"、"绞痛"
     - 例 trigger 答"既吃了红烧肉,又熬夜" → slot_fills.trigger=["进食油腻","熬夜"](多值 list) + 不抽症状(诱因不是症状)
     - 例 trigger 答"没明显诱因" → slot_fills.trigger=["(患者未明确)"] + 不抽症状(只是"未明确",不是症状信号)
     - **跳过 `associated_symptoms` slot**:它本身就是症状清单(已在 slot_fills 里),**不要重复**写到 confirmed_symptoms,避免重叠{obstetric_rule}{history_rule}

# 输出格式(严格 JSON,**字段名必须照下方模板原样,不要新造字段名,不要 markdown 代码块包裹,不要解释**)

```
{{
  "slot_fills": {{
    "<HPI 12 维之一的英文槽名,如 onset_time / location / aggravating>": "<单值槽 str 或多值槽 list[str]>"
  }},
  "confirmed_symptoms": ["<患者明确确认的症状,如 '恶心'>", "..."],
  "denied_symptoms": ["<患者明确否认的症状,如 '呕吐'>", "..."],
  "uncertain_symptoms": ["<患者模糊语气提及的症状,如 '头晕'>", "..."],
  "obstetric_fills": {{
    "is_pregnant": true,
    "is_lactating": false
  }},
  "history_fills": {{
    "allergies": ["<过敏原名>"],
    "medications": ["<药物名>"],
    "past_conditions": ["<既往疾病名>"]
  }}
}}
```

注意:
- `obstetric_fills` 整段仅当本轮含妊娠/哺乳追问才出现,否则**整个字段省略**(不要写 null,不要写 {{}})
- `history_fills` 整段仅当本轮含病史追问才出现,否则**整个字段省略**
- `slot_fills` / `confirmed_symptoms` / `denied_symptoms` / `uncertain_symptoms` 这四个字段**始终出现**,无内容时写 {{}} / [] 而非省略
"""


# ────────────────────────────────────────────────────────────────────────────
# intake_followup_ask LLM 针对性阶段(12 维 slot 全填后调一次)
# ────────────────────────────────────────────────────────────────────────────


def build_targeted_followup_prompt(
    chief_complaint: str,
    present_illness: str,
    filled_slots: dict,
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    uncertain_symptoms: list[str],
    medical_history_summary: str,
    quota: int,
) -> str:
    """⑤ Step A(flash 决策)prompt:基于已填 HPI + 病史出**决策**双 list。

    - askable_targets:**中文短语**,患者主观能告诉的靶点(如"疼痛放射到背部")
      → 后续 Step B(flash)再把每个 target 转成自然中文问句
    - unaskable_findings:患者答不上,必须查体/化验/影像才能知道的客观证据
      → ⑧a 首诊推单(已是医生侧 description + reason,无需再加工)

    2026-05-22:askable_target 从英文 snake_case 改中文短语,跟 denied/confirmed
    中文列表同语种,直接字符串对账去重,flash 不再"心里翻译"。

    严格约束:**只判该问/该查,不诊断**;**不要再问已知信息**(prompt 内显式去重要求)。
    """
    slots_block = "\n".join(f"  - {k}:{_format_slot_value(v)}" for k, v in filled_slots.items()) or "  (无)"
    confirmed_block = "、".join(confirmed_symptoms) or "(无)"
    denied_block = "、".join(denied_symptoms) or "(无)"
    uncertain_block = "、".join(uncertain_symptoms) or "(无)"

    return f"""你是问诊 holistic gate,**只负责判"还差什么信息",绝不做诊断**。
你的输出会按"患者能不能答"分两路:
  - 患者主观能答的 → 出**中文短语**(askable_targets),稍后由另一个 LLM 拼成自然问句
  - 患者答不上,必须查体/化验/影像才能知道的 → 让用户去医院做(unaskable_findings)

【当前已知】
- 主诉:{chief_complaint}
- 现病史:{present_illness}
- HPI 12 维(已填部分):
{slots_block}
  (若某 value 为 "(患者未明确)" 或列表含此值:**已问过但患者答不上**,**不要再对该维度追问**;
   但可以把它列进 unaskable_findings,让 ⑧a 推查体/检查去拿这个信息)
- 已确认症状:{confirmed_block}
- 已否认症状:{denied_block}
- 不确定症状:{uncertain_block}
- 病史档案摘要:
{medical_history_summary}

【你的任务】
判断进入检索/诊断前还差什么信息,按"患者能不能答"分两路:

1. **askable_targets**(0~{quota - 1} 条):患者用语言能告诉你的主观信息**靶点**
   - **只输出中文短语**(简洁名词性短语,不写完整问句!不写英文!)
   - 例:"发烧最高温度" / "疼痛放射到背部" / "饭后是否加重" / "皮肤巩膜黄染" /
     "既往类似腹痛史" / "胆石家族史" / "近期饮酒" / "疼痛 NRS 评分"
   - 完整自然问句由下游 LLM(flash)拼,你只负责"该问哪些靶点"
   - **专业医学缩写允许英文**(NRS / WBC / Murphy 等),其他必须中文

2. **unaskable_findings**(0~{quota} 条):患者答不上,需要查体或检查才能确定的客观证据
   - 例:{{"description": "腹部 Murphy 征查体", "reason": "鉴别胆囊炎 vs 胃炎"}}
   - 例:{{"description": "血常规 + 肝功能", "reason": "判断有无感染 / 肝胆受累"}}
   - 例:{{"description": "腹部 B 超", "reason": "看胆囊壁厚度、有无结石"}}
   - description = 医生侧语言,写"查什么"(下游 ⑧a 会转译成患者友好文案);reason = 为什么对鉴别重要

【严格约束(违反即视为输出失败)】
1. 禁止输出疾病名/诊断/可能性/probability/differential 等任何诊断性词汇
2. 禁止做"我怀疑是 X"/"考虑 Y"/"可能性较高"的判断
3. askable 只列**主观表现**靶点(发热温度、放射部位、加重缓解、伴随表现等)
4. unaskable 只列**客观证据**(查体/化验/影像/心电图等);**不要把"问患者有没有"放进 unaskable**
5. 两路不要重复同一个信息(同一项要么 askable 要么 unaskable,不能两边都出)
6. **去重铁律**:已经出现在【当前已知】任意位置的信息**绝不再问**(因 askable_target 现在
   是中文短语,直接字符串包含/语义重叠就算"已在列表里"):
   - 已填 HPI 维度已经有值的(包括"已问过但患者答不上"的)→ 不要重复
   - 已确认/已否认/不确定症状列表里的症状名 → 不要再问"有没有 X" 也不出 X 的细分变体
   - 病史档案显示"已询问,无"的范畴(过敏/慢病/既往疾病/家族史)→ 不要再问该范畴
   - 现病史自由文本已经提到的细节(如温度数值、伴随表现)→ 不要再问相同细节
   - 例:已 denied 含"胆石家族史" → askable 不再出"胆石家族史"或"父母兄弟胆结石史"
   - 例:已 denied 含"近期饮酒" → askable 不再出"发作前饮酒"等细分(完全否认已 cover)
7. 信息已充分时,两个 list 都输出 [] — 不要硬凑

# 输出格式(严格 JSON,**字段名必须照下方模板原样,不要新造字段名,不要 markdown 代码块包裹,不要解释**)

```
{{
  "askable_targets": ["<中文短语,如 '发烧最高温度'>", "<...>"],
  "unaskable_findings": [
    {{"description": "<医生侧语言,如 '腹部 Murphy 征查体'>", "reason": "<为什么对鉴别重要,如 '鉴别胆囊炎 vs 胃炎'>"}}
  ]
}}
```

注意:
- 两个字段**始终出现**,无内容时写 [] 而非省略
- `askable_targets` 数组上限 {quota - 1} 条;`unaskable_findings` 数组上限 {quota} 条
"""


# ────────────────────────────────────────────────────────────────────────────
# ⑤ Step B(flash 拼问句)— askable_targets → 自然中文问句
# ────────────────────────────────────────────────────────────────────────────


def build_question_generation_prompt(
    chief_complaint: str,
    present_illness: str,
    askable_targets: list[str],
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
) -> str:
    """⑤ Step B prompt:把 Step A 出的中文短语 list 转成患者侧自然中文问句。

    用 flash(非 thinking)跑,耗时 1-3s。输出 1-1 映射:每个 target 出一条
    {question, target},target 必须原样回填(中文短语,便于下轮去重)。

    2026-05-22:
      - askable_target 从英文 snake_case 改中文短语(全链路中文统一)
      - 增加 confirmed/denied 上下文,prompt 约束"串联小问不能带入已知症状"
        (修问题 6:Step B 把已 denied 的"寒战""尿色加深"打包进问句导致半重复)
    """
    targets_block = "\n".join(f"  - {t}" for t in askable_targets)
    confirmed_block = "、".join(confirmed_symptoms) or "(无)"
    denied_block = "、".join(denied_symptoms) or "(无)"
    return f"""你是问诊文案助手,把医生侧的中文靶点短语转成患者能听懂的自然中文问句。

【上下文】
- 主诉:{chief_complaint}
- 现病史:{present_illness}
- 已确认症状(不要再问"有没有 X"):{confirmed_block}
- 已否认症状(**串联打包时绝对不能再带入这些**):{denied_block}

【要转换的靶点列表(中文短语)】
{targets_block}

【转换规则】
1. 每个 target 转成 **1 条 question + target 原样回填(中文短语,不要翻译成英文)**
2. question 是**自然中文口语**问句,患者能听懂,可以串联同主题的小问 —— 但**串联内容必须
   全是新的鉴别点**,**严禁带入【已确认/已否认】列表里出现过的症状词**:
   - 例 target="发烧最高温度" + denied 含"寒战" → "您发烧最高到多少度?"(不能再串"有没有寒战")
   - 例 target="发烧最高温度" + denied 不含"寒战" → "您发烧最高到多少度?有没有寒战?"(可串)
   - 例 target="排便异常" + denied 含"大便颜色变浅" → "您最近排便习惯有变化吗,比如便秘或腹泻?"(剔除"颜色")
   - 例 target="小便异常" + denied 含"尿色加深" → "您有没有尿频、尿急或尿痛?"(剔除"尿色")
3. **严禁泛化代词**(如"类似情况"/"相同症状"/"这种问题"/"上面那个"等),
   必须把**主诉的具体症状名嵌入问句**,让下游(⑦ flash 解析)和患者都能明确知道指什么:
   - 例 target="既往类似腹痛史" + 主诉"右上腹疼痛" → "您**以前有没有过类似的右上腹疼痛**?"
     **不**写 "您以前有过类似情况吗?"
   - 例 target="症状演变趋势" + 主诉"上腹痛" → "您的**上腹痛**是越来越重还是越来越轻?"
     **不**写 "症状有变化吗?"
   - **关键**:把主诉/现病史里出现过的症状词(如"右上腹疼痛""发热""恶心")**原样嵌进 question**,
     这样患者一眼看懂问的是什么,⑦ 解析时也能从 question 抽出靶点症状
4. **不要诊断、不要给疾病名**,只问表现
5. 顺序与输入顺序保持一致;target 字段**原样**回填输入的中文短语(不改写、不翻译、不删字)

# 输出格式(严格 JSON,**字段名必须照下方模板原样,不要 markdown 代码块包裹,不要解释**)

```
{{
  "questions": [
    {{"question": "<患者侧自然中文问句>", "target": "<原样回填输入的中文短语>"}}
  ]
}}
```
"""


# ────────────────────────────────────────────────────────────────────────────
# ⑧a recommend_exam(自由文本输出)
# ────────────────────────────────────────────────────────────────────────────


def build_recommend_exam_prompt(
    *,
    mode: str,
    chief_complaint: str,
    present_illness: str,
    diagnosis_results: list[dict],
    unaskable_symptoms: list[dict],
    existing_report_findings: list[dict],
) -> str:
    """⑧a recommend_exam — **双模式**:

    - mode="intake":⑤ 触发,无 diagnosis_result。基于主诉/HPI + `unaskable_symptoms`
      推首诊全套(让患者一次性查齐,⑩ 大概率一次结案)
    - mode="differential":⑩ 后 need_exam,有 diagnosis_result + 精筛的 retained_unaskable。
      基于候选疾病推针对性补漏

    **不再读 candidate_chunks** — 医学推理已在 ⑤/⑩ 完成,⑧a 只做"医生侧 description →
    患者友好文案 + 优先级排序",prompt 短延迟低。

    `unaskable_symptoms` 在两种模式下都是主源:首诊模式来自 ⑤,鉴别模式来自 ⑩ 精筛覆盖。
    """
    unaskable_lines = [
        f"  - {u.get('description')} —— {u.get('reason')}"
        for u in unaskable_symptoms[:8]
    ]
    unaskable_block = "\n".join(unaskable_lines) or "  (无)"

    existing_lines = []
    for r in existing_report_findings[:5]:
        existing_lines.append(
            f"  - {r.get('report_type')} ({r.get('report_date')}): "
            f"impressions={(r.get('impressions') or [])[:2]}"
        )
    existing_block = "\n".join(existing_lines) or "  (无)"

    if mode == "intake":
        # 首诊模式:无诊断假设,基于主诉 + HPI + ⑤ 写的 unaskable 推全套
        context_block = f"""【模式】首诊模式 — 患者尚未诊断,需推荐"为鉴别清楚最可能的几类病所需的标准首查清单"

【患者主诉 + 现病史】
- 主诉:{chief_complaint}
- 现病史:{present_illness}

【⑤ holistic gate 标记的不可问发现(主源,直接转检查清单)】
{unaskable_block}"""
        task_block = """【任务】
基于主诉 + ⑤ 已标记的 unaskable 清单,推荐 3-5 项检查/查体(按优先级)。
**直接消费 unaskable_findings 的 description**(它已经写好"该查什么")作为主要候选;
如有遗漏的常规鉴别检查(常见主诉的标准首查),酌情补充 1-2 项。"""
    else:
        # 鉴别模式:有候选疾病,基于 retained_unaskable 推补漏
        diag_lines = [
            f"  - {r.get('disease')} (p={r.get('probability', 0):.2f}, type={r.get('differentiation_type')})"
            for r in diagnosis_results[:5]
        ]
        diag_block = "\n".join(diag_lines) or "  (无诊断结果)"
        context_block = f"""【模式】鉴别模式 — ⑩ 已诊断但仍需检查补漏,基于候选疾病推针对性鉴别

【诊断候选】
{diag_block}

【⑩ 精筛的需鉴别体征(retained_unaskable)】
{unaskable_block}"""
        task_block = """【任务】
基于候选疾病的鉴别要点 + retained_unaskable,推荐 3-5 项检查(按优先级),
每项说明"区分什么"。"""

    return f"""你是医生检查建议助手。请按下方模式 + 上下文,推荐检查项给患者,**按"医院实际出报告的载体"分组**。

{context_block}

【患者已有报告】
{existing_block}

{task_block}

【分组规则 — 按"医院实际出报告的载体"分,不是按"检查项的医学分类"】

医院的检查结果不是按项目 1:1 出报告:一次抽血出多个指标(一张化验单);
体格检查写在病历本上(一页);影像各自独立报告。所以分组要贴合患者拿到的实际载体:

  1. **抽血化验类**(一次抽血出多个指标,**合 1 组**,医院给一张化验单或一组打印件)
     - 例:血常规 / 肝功能 / 肾功能 / 电解质 / CRP / PCT /
       淀粉酶 / 脂肪酶 / 心肌酶 / 转氨酶 / 凝血 / 血型
     - group_label 例:"抽血化验(空腹8h)" / "抽血化验(急查)"
     - note 例:"医院一次抽血出多个指标,一张化验单或一组报告"

  2. **体格检查类**(医生当场判断写在病历本/急诊记录,**合 1 组**)
     - 例:Murphy 征 / 压痛 / 反跳痛 / 肌紧张 / 心音听诊 / 神经反射
     - group_label 例:"医生体格检查记录"
     - note 例:"医生写在门诊病历本或急诊记录上,拍病历对应页即可"

  3. **影像类**(**各自独立报告**,不合并)
     - 例:腹部 B 超 / 胸部 X 光 / 头颅 CT / 腹部 MRI / 胃肠造影
     - group_label 直接用检查名,如 "腹部 B 超(肝胆胰脾)"
     - note 例:"需空腹 6-8 小时" / "报告单 + 影像描述"

  4. **其他独立报告**(各自独立)
     - 心电图 / 病理 / 胃镜 / 肠镜 / 肺功能
     - group_label 直接用检查名

不确定能不能合并 → **单独成组**(宁多勿少,避免患者混淆)。一组 items 不超过 6 条。

【输出要求】
- `test_groups`: list,典型 2-5 组(覆盖 4 大类),按优先级排序(最关键的在前)
- 每组 `group_label`(简洁中文组名)+ `items`(检查项 list[str])+ `note`(给患者的提示,1-2 句)
- 对与已有报告交集的检查,追加"已有[日期]报告,可携带评估是否需要复做"到 note,不要静默删除
- 口吻面向患者,避免医学术语堆砌,涉及禁食/造影剂等特殊条件写到 note
- `rationale`: str,整体说明(2-3 句,为什么这几组对当前主诉/鉴别最关键)""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ⑩ diagnose 1 步 prompt(对齐评测口径 .eval/rag_eval/run_diagnose_eval.py)
# ────────────────────────────────────────────────────────────────────────────


def _format_medical_history(history: dict) -> str:
    """把 8 张患者子表 dict 格式化成结构化中文段落,供 ⑩ diagnose prompt 用。

    设计原则(替代 json.dumps()[:600] 暴力截断):
    - 每个子项独立一行,前缀 `· {中文名}:`
    - 非空 → 列出具体内容
    - 空 list / 空 dict / None → "(未询问 — 不代表阴性)"
    - **绝不截断**:1M context 全给 LLM 看,对齐评测口径
    - **空字段语义:未询问 ≠ 阴性**(当前 schema 无 is_denied 字段,
      患者明确否认的信息暂时只能进 raw_text;LLM 应理解空字段为"信息缺失")
    """
    if not history:
        return "(档案为空 — 患者首诊或未填档,所有维度均未询问,不代表阴性)"

    lines: list[str] = []

    # 既往疾病(嵌在 past_history 子 dict 里)
    past = history.get("past_history") or {}
    med_list = past.get("medical_history") or []
    if med_list:
        items = []
        for r in med_list:
            seg = r.get("condition") or "(未命名)"
            extras = []
            if r.get("diagnosed_at"):
                extras.append(f"诊断{r['diagnosed_at']}")
            if r.get("control_status"):
                extras.append(f"控制:{r['control_status']}")
            if r.get("notes"):
                extras.append(r["notes"])
            if extras:
                seg += f"({'; '.join(extras)})"
            items.append(seg)
        lines.append("· 既往疾病:" + "; ".join(items))
    else:
        lines.append("· 既往疾病:(未询问 — 不代表阴性)")

    # 手术/外伤史
    surg = past.get("surgical_trauma") or []
    if surg:
        items = []
        for r in surg:
            seg = r.get("name") or "(未命名)"
            if r.get("occurred_at"):
                seg += f"({r['occurred_at']})"
            if r.get("sequelae"):
                seg += f"[后遗:{r['sequelae']}]"
            items.append(seg)
        lines.append("· 手术/外伤史:" + "; ".join(items))
    else:
        lines.append("· 手术/外伤史:(未询问 — 不代表阴性)")

    # 输血史
    trans = past.get("transfusion") or []
    if trans:
        items = []
        for r in trans:
            seg = r.get("blood_product") or "(未注明)"
            if r.get("transfusion_date"):
                seg += f"({r['transfusion_date']})"
            if r.get("adverse_reaction"):
                seg += f"[不良反应:{r.get('reaction_detail') or '有'}]"
            items.append(seg)
        lines.append("· 输血史:" + "; ".join(items))
    else:
        lines.append("· 输血史:(未询问 — 不代表阴性)")

    # 过敏史(安全敏感)
    allergies = history.get("allergy_history") or []
    if allergies:
        items = []
        for r in allergies:
            seg = r.get("substance") or "(未命名)"
            if r.get("reaction"):
                seg += f"→ {r['reaction']}"
            if r.get("severity"):
                seg += f"[{r['severity']}]"
            items.append(seg)
        lines.append("· 过敏史 ⚠️ :" + "; ".join(items))
    else:
        lines.append("· 过敏史 ⚠️ :(未询问 — 不代表阴性,safety_gate 无法兜底)")

    # 用药史(safety_gate 用)
    meds = history.get("medication_history") or []
    if meds:
        current_items, past_items = [], []
        for r in meds:
            seg = r.get("drug_name") or "(未命名)"
            if r.get("dosage") or r.get("frequency"):
                seg += f"({r.get('dosage') or ''} {r.get('frequency') or ''})".strip()
            if r.get("is_current"):
                current_items.append(seg)
            else:
                past_items.append(seg)
        parts = []
        if current_items:
            parts.append("当前服用:" + "; ".join(current_items))
        if past_items:
            parts.append("既往用药:" + "; ".join(past_items))
        lines.append("· 用药史 ⚠️ :" + " | ".join(parts))
    else:
        lines.append("· 用药史 ⚠️ :(未询问 — 不代表未服药)")

    # 个人史(吸烟/饮酒/职业/暴露)— 内嵌在 patients 表
    personal = history.get("personal_history") or {}
    if personal:
        bits = []
        if personal.get("smoking_status"):
            seg = f"吸烟:{personal['smoking_status']}"
            if personal.get("smoking_pack_years"):
                seg += f"({personal['smoking_pack_years']} 包年)"
            bits.append(seg)
        if personal.get("alcohol_status"):
            seg = f"饮酒:{personal['alcohol_status']}"
            if personal.get("alcohol_detail"):
                seg += f"({personal['alcohol_detail']})"
            bits.append(seg)
        if personal.get("occupation"):
            bits.append(f"职业:{personal['occupation']}")
        if personal.get("occupational_exposure"):
            bits.append(f"职业暴露:{personal['occupational_exposure']}")
        if personal.get("travel_history"):
            bits.append(f"旅居史:{personal['travel_history']}")
        if personal.get("infectious_contact"):
            bits.append(f"传染病接触:{personal['infectious_contact']}")
        lines.append("· 个人史:" + (" | ".join(bits) if bits else "(字段全空)"))
    else:
        lines.append("· 个人史:(未询问 — 不代表阴性)")

    # 婚育/月经史(女性)
    ob = history.get("obstetric_history")
    if ob:
        bits = []
        if ob.get("pregnancy_status"):
            bits.append(f"孕:{ob['pregnancy_status']}")
        if ob.get("lactation_status"):
            bits.append(f"哺乳:{ob['lactation_status']}")
        if ob.get("gravidity") is not None or ob.get("parity") is not None:
            bits.append(f"孕产:G{ob.get('gravidity', '?')}P{ob.get('parity', '?')}")
        if ob.get("last_menstrual_period"):
            bits.append(f"末次月经:{ob['last_menstrual_period']}")
        if ob.get("menopause_age"):
            bits.append(f"绝经年龄:{ob['menopause_age']}")
        if ob.get("notes"):
            bits.append(ob["notes"])
        lines.append("· 婚育/月经史:" + (" | ".join(bits) if bits else "(字段全空)"))
    else:
        lines.append("· 婚育/月经史:(未询问 — 男性可忽略;女性视情况补问)")

    # 家族史
    family = history.get("family_history") or []
    if family:
        items = []
        for r in family:
            seg = f"{r.get('relation', '亲属')}:{r.get('condition', '未命名')}"
            if r.get("onset_age"):
                seg += f"(发病{r['onset_age']}岁)"
            items.append(seg)
        lines.append("· 家族史:" + "; ".join(items))
    else:
        lines.append("· 家族史:(未询问 — 不代表阴性)")

    return "\n".join(lines)


def build_diagnose_prompt(
    *,
    parent_texts: list[str],
    figures: list[dict],
    chief_complaint: str,
    present_illness: str,
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    uncertain_symptoms: list[str],
    slots: dict[str, Any],
    medical_history: dict,
    report_findings: list[dict],
    unaskable_symptoms: list[dict],
) -> tuple[list[BaseMessage], str]:
    """⑩ diagnose 1 步 LLM prompt(多模态,对齐评测口径 .eval/rag_eval/run_diagnose_eval.py)。

    返回多模态 messages + 纯文本 prompt(后者供 §9.6 final_prompt 审计存档)。
    figure 的 image_data_uri 作为 image_url 消息块附加;medical_statement 已在
    context builder 中排除,**不进 prompt**(spec §3.1.5.1 + §3.2.3 关键认知)。

    Args:
        parent_texts: Step 0.5 父块扩展后的文本列表(与 reranked_chunks 同序)
        figures: Step 0.5 去重后的图表 chunk 列表,每条含 chunk_raw_text + image_data_uri
        chief_complaint / present_illness: 患者叙事(原文)
        confirmed_symptoms / denied_symptoms / uncertain_symptoms / slots /
            medical_history / report_findings: 多轮 followup 累积的患者画像
            (medical_history 由 _format_medical_history 结构化展开,空字段显式标"未询问")
        unaskable_symptoms: ④ 写入的粗筛版({description, reason}),供 LLM 产 retained_unaskable
    """
    # 父块文本:对齐评测口径,不截断(LLM 1M context,信息全给)
    parents_block = "\n\n".join(
        f"[文本块 {i+1}]\n{(c or '')}" for i, c in enumerate(parent_texts)
    ) or "(无召回)"

    if figures:
        figures_caption_block = "【召回 figure 截图(随附图像消息块,按下面顺序看)】\n" + "\n".join(
            f"[figure {i+1} | {f.get('chunk_type')}] {(f.get('chunk_raw_text') or '')[:300]}"
            for i, f in enumerate(figures)
        )
    else:
        figures_caption_block = "【召回 figure 截图】(无)"

    confirmed_block = "、".join(confirmed_symptoms) or "(无)"
    denied_block = "、".join(denied_symptoms) or "(无)"
    uncertain_block = "、".join(uncertain_symptoms) or "(无)"
    slots_block = json.dumps({k: v for k, v in slots.items() if v}, ensure_ascii=False)
    history_block = _format_medical_history(medical_history)
    reports_block = json.dumps(report_findings[:5], ensure_ascii=False)[:1500]
    unaskable_block = json.dumps(unaskable_symptoms[:8], ensure_ascii=False) if unaskable_symptoms else "[]"

    prompt_text = f"""你是临床鉴别诊断助手。基于以下患者信息 + 检索召回的医学文献(含父块全文 + table HTML
+ 可选 figure 截图),做鉴别诊断并按概率降序输出候选疾病。

【患者主诉】
{chief_complaint or "(无)"}

【现病史原文】
{present_illness or "(无)"}

【现病史结构化维度】
{slots_block}
(若某维度 value 为 "(患者未明确)" 或列表含此值:**已问过但患者答不上**;
 视同"患者明确否认/不知",**不作为阳性证据**,可在 differentiation 里指出"此维度信息缺失,影响鉴别")

【多轮交互累积的患者画像】
- 已确认症状:{confirmed_block}
- 已否认症状:{denied_block}
- 已问但不确定的症状:{uncertain_block}

【病史档案(空字段 = 尚未询问,不代表阴性 — 请在 evidence/differentiation 中酌情指出待补)】
{history_block}

【检查报告发现】
{reports_block}

【上游写入的 unaskable 粗筛(④ 鉴别诊断 + ⑤ 检索前 holistic 累积,供 retained_unaskable 精筛参考)】
{unaskable_block}

【医学文献文本(RAG 召回 Top-{len(parent_texts)} 父块 + table HTML,按相关性顺序)】
{parents_block}

{figures_caption_block}

【任务】
1. 列出候选疾病(至少 1 个,通常 1-5 个),按 probability 降序输出
2. 每个候选给出:
   - disease:疾病名(精确到部位/分型,如 "右额颞急性硬膜外血肿" 而非 "颅内血肿")
   - probability:0~1 的概率(每个候选独立估算,不需归一)
   - evidence:3-5 条关键支持证据(可引用症状/报告/文献/图像)
   - differentiation:与其他相似疾病的鉴别要点(可空)
   - differentiation_type(必出):
     * `confirmed` — top1 概率 ≥ 0.6 且证据闭环
     * `need_exam` — top1 概率 0.3-0.6,或多个候选概率接近(差距 < 0.1),鉴别依赖检查体征
     * `insufficient` — top1 概率 < 0.3,或候选分散证据不足支持任何高概率判断
     * **top1 决定后续路由**:`need_exam` → 走 ⑧ recommend_exam;其他 → 走 ⑪ safety_gate;
       top2/top3 沿用 top1 的值即可(router 只看 top1)
   - failure_reason:**保持 null**(由节点代码在兜底路径填,不在 LLM 职责范围)
3. retained_unaskable(主要从【上游 unaskable 粗筛】里精筛 + 改写,**也允许新产**):
   - top1=`confirmed` → 通常返空列表(证据已闭环,无需再查)
   - top1=`insufficient` → 通常返空列表(检查也救不回信息不足)
   - top1=`need_exam` → **至少保留 1 条**,只留对当前 top 候选鉴别真正关键的;描述可改写
     得更聚焦,如把"想知道胆囊有无问题"改成"腹部 B 超确认胆囊壁厚度 + 有无结石"
   - **允许新产**:诊断推理后觉得"上游没列但鉴别真需要"的检查项(如 MRCP / 特定肿瘤标志物等),
     可以直接加进 retained_unaskable;但**不要为加而加** — 大多数 case 挑/改写就够
   - **宁可少留不可多留** — 不该查的留下来会被 ⑧a 直接推给患者""" + _JSON_TAIL

    # 多模态消息组装:base text + 每张可加载的 figure 截图作 image_url 块
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for f in figures:
        uri = f.get("image_data_uri")
        if uri:
            content.append({"type": "image_url", "image_url": {"url": uri}})

    if len(content) == 1:
        # 没图就直接 str content,避免 provider 把 list 当多模态特殊处理
        messages: list[BaseMessage] = [HumanMessage(content=prompt_text)]
    else:
        messages = [HumanMessage(content=content)]

    return messages, prompt_text


# ────────────────────────────────────────────────────────────────────────────
# ⑪ safety_gate(LLM 兜底层)
# ────────────────────────────────────────────────────────────────────────────


def build_safety_gate_prompt(
    diagnosis_results: list[dict],
    medical_history: dict,
    rule_layer_constraints: dict,
) -> str:
    """⑪ LLM 兜底层:在规则层基础上识别交叉过敏 / 罕见相互作用 / 肝肾剂量调整。"""
    diag_block = json.dumps(diagnosis_results[:3], ensure_ascii=False)
    history_block = json.dumps(medical_history, ensure_ascii=False)
    rules_block = json.dumps(rule_layer_constraints, ensure_ascii=False)

    return f"""你是临床用药安全助手。下面是规则层已完成的安全约束,请在此基础上做**LLM 兜底**——
识别规则层未覆盖的额外风险。

【诊断结果】{diag_block}
【病史】{history_block}
【规则层约束】{rules_block}

【兜底任务(只输出 additional_risks)】
- cross_allergy:交叉过敏风险,如对头孢过敏 → 警告青霉素类
- interaction:罕见或新近发现的药物相互作用
- dosage_adjustment:基于肝肾功能的剂量调整(从病史推断)

每项含 risk_type / description / severity(high/medium/low) / recommendation。
若无新风险,additional_risks 留空列表。**不要重复规则层已写的禁忌**。""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ⑫ generate_advice
# ────────────────────────────────────────────────────────────────────────────


def build_advice_prompt(
    diagnosis_results: list[dict],
    safety_constraints: dict,
    failure_reason: str | None,
) -> str:
    """⑫ 在 safety_constraints 约束内生成用药/检查/风险建议。"""
    diag_block = json.dumps(diagnosis_results[:3], ensure_ascii=False)
    safety_block = json.dumps(safety_constraints, ensure_ascii=False)
    failure_note = f"\n【系统失败提示】{failure_reason}" if failure_reason else ""

    return f"""你是医生治疗建议助手。基于诊断 + 安全约束,产出结构化建议。

【诊断结果】{diag_block}
【安全约束】{safety_block}{failure_note}

【输出规则】
- medications:用药建议,每项 drug_name / dosage / frequency / duration / notes
  ,所有药物**必须不在 banned_drugs 列表里**,且不触发 interaction_warnings
- exam_suggestions:建议检查项目(基于 differentiation_type=='need_exam' 的候选 +
  插入安全约束相关的功能监测)
- risk_warnings:风险提示与注意事项,**包含**:
  - 高危场景警告(如疑似心梗/脑卒中)
  - safety_constraints.contraindication_flags 的患者侧解释
  - 系统失败提示对应的患者侧告知(如 followup_round_capped → "建议线下就诊获得
    更全面评估";step_N_failed → "系统分析出现技术问题,本次结果不可作为依据")
- urgent_flag:疑似心梗/脑卒中/消化道大出血等高危情况 → True

口吻面向普通患者,不要堆砌术语。""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ⑬ format_response(自由文本)
# ────────────────────────────────────────────────────────────────────────────


def build_format_response_prompt(
    diagnosis_results: list[dict],
    medication_advice: list[dict],
    recommended_tests: list[str],
    risk_warnings: list[str],
    failure_reason: str | None,
) -> str:
    """⑬ 自由文本最终回复:整合诊断 + 建议 + 免责声明。"""
    diag_block = json.dumps(diagnosis_results[:3], ensure_ascii=False)
    med_block = json.dumps(medication_advice, ensure_ascii=False)
    tests_block = json.dumps(recommended_tests, ensure_ascii=False)
    risk_block = json.dumps(risk_warnings, ensure_ascii=False)

    failure_disclaimer = ""
    if failure_reason:
        failure_disclaimer = (
            "\n本次诊断因系统原因未能完整推理,结果仅供参考,请务必线下就诊。"
        )

    return f"""你是医院分诊台问诊助手。请把下列结构化结果整合成一段**患者可读**的自然语言回复。

【诊断】{diag_block}
【用药】{med_block}
【建议检查】{tests_block}
【风险提示】{risk_block}

【回复结构】
1. 一段简短诊断说明(候选疾病 + 大致可能性,口语化,不直接报概率数字)
2. 用药 / 检查 / 注意事项,分点说明
3. 风险提示(若有 urgent_flag → 强烈建议立即就医放到最前)
4. 免责声明:本结果仅作分诊参考,不代替线下医生诊断;具体方案请咨询执业医师{failure_disclaimer}

整段控制在 200-400 字。"""
