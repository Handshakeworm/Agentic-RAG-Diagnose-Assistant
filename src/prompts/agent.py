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


# ────────────────────────────────────────────────────────────────────────────
# ① info_collect Step 1
# ────────────────────────────────────────────────────────────────────────────


def build_info_collect_prompt(patient_input: str) -> str:
    """① info_collect Step 1:从 patient_input 提取主诉 + 现病史 + 13 维槽位。

    输出由 `InfoCollectOutput` schema 严格约束,LLM 只需聚焦"提取什么、不
    提取什么"。
    """
    return f"""你是医院分诊台问诊助手。下面是患者的自述。请仅从该自述中提取本次就诊的信息,
**不要自行编造、不要泛化、不要补充未提及的内容**。

【患者自述】
{patient_input}

【提取要求】
1. chief_complaint(主诉):主要症状 + 持续时间,1 句话,例:"腹痛3天"
2. present_illness(现病史):用 1-3 句话展开本次发病的:起病时间、诱因、症状特点
   (部位/性质/程度)、伴随症状、加重/缓解因素、治疗经过
3. present_illness_slots(13 维结构化槽位):
   - 单值槽(str | None):onset_time / onset_mode / trigger / location / nature /
     severity / duration_pattern / progression / treatment_tried / treatment_response
   - 多值槽(list[str]):aggravating(加重因素) / relieving(缓解因素) /
     associated_symptoms(伴随症状)
   - 患者**未提及**的维度严格保持 None / 空列表,**不要瞎填**

注意:这是初诊采集,信息缺失是正常的,后续会通过追问补全。""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ①.5 analyze_initial_reports / ⑨ process_exam_result
# ────────────────────────────────────────────────────────────────────────────


def build_report_parsing_prompt(num_reports: int) -> str:
    """①.5 / ⑨ 多模态 LLM 直读报告 → 结构化关键发现。

    Args:
        num_reports: 本次解析的报告数量,prompt 中提示 LLM 输出对应数量的 finding 项

    多模态附件(图片 base64 / PDF 文件)由调用方组装到 LangChain message 中,
    本函数只产文本提示部分。
    """
    return f"""你是医学报告结构化解析助手。下面附了 {num_reports} 份检查报告(图片或 PDF)。
请逐份解析,产出结构化的 ReportFindings 列表,每份报告对应 findings 列表中的一项。

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
- negation:是否被否定。如"没有发烧" → True;"发烧" → False
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
    slots_lines = [f"  - {k}: {v}" for k, v in filled_slots.items() if v]
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
# ⑤ select_discriminative_symptom — Smart followup(1 LLM)
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
    """⑤ 1 次 LLM 同时出 questions(追问) + unaskable_symptoms(粗筛)。"""
    filled_lines = [f"  - {k}: {v}" for k, v in filled_slots.items() if v]
    filled_block = "\n".join(filled_lines) if filled_lines else "  (无,全部空缺)"
    empty_block = ", ".join(empty_slots) or "(无,13 维已全部填满)"
    conf_block = "、".join(confirmed_symptoms) or "(无)"
    den_block = "、".join(denied_symptoms) or "(无)"
    unc_block = "、".join(uncertain_symptoms) or "(无)"

    return f"""你是临床问诊助手。请基于患者已提供的信息,同时产出本轮**追问项** + **想知道但患者答不上的体征**。

【患者主诉】{chief_complaint or "(无)"}
【现病史描述】{present_illness or "(无)"}

【已填的 HPI 维度】
{filled_block}

【空缺的 HPI 维度】(13 维框架内尚未问到的)
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
   - 适合用在:13 维已大部分填满 / 空缺维度都不重要 / 想兜底捕获遗漏症状
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
# ⑥a generate_followup
# ────────────────────────────────────────────────────────────────────────────


def build_followup_question_prompt(
    chief_complaint: str,
    questions: list[dict],
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
) -> str:
    """⑥a 生成两种 type(slot 维度填补 + open 开放式)追问问题,患者口语风格。"""
    items = []
    for q in questions:
        if q.get("type") == "slot":
            items.append(f"  - 补全 HPI 维度:{q['slot']}")
        elif q.get("type") == "open":
            items.append("  - 开放式问:还有没有别的地方不舒服")
    items_block = "\n".join(items) if items else "  (无)"

    confirmed_block = "、".join(confirmed_symptoms) or "(无)"
    denied_block = "、".join(denied_symptoms) or "(无)"

    return f"""你是问诊助手。患者主诉:"{chief_complaint}"。
已确认有的症状:{confirmed_block}
已否认的症状:{denied_block}

请把下列待追问项**自然合并成 2-3 句**患者口语化的追问。**不要列举式**,不要"问题1/问题2",
要像聊天一样自然过渡。

【待追问项】
{items_block}

输出要求:
- 直接给问题文本,不要前缀"请问"反复出现
- 维度补全(slot):用"是什么情况下/怎样的/最近有没有变化"等口语表达,不要直接说"诱因/性质"
  这类医学术语
- 开放式追问(open):自然问"除了上面说的,还有没有别的地方不舒服?" — 用于兜底捕获遗漏症状
- 控制在 2-3 句以内
- 涉及隐私/心理症状要用委婉表达"""


# ────────────────────────────────────────────────────────────────────────────
# ⑦ process_followup_answer
# ────────────────────────────────────────────────────────────────────────────


def build_followup_parse_prompt(
    followup_question: str,
    followup_answer: str,
    questions: list[dict],
) -> str:
    """⑦ 解析患者回答 → 维度槽位回填 + 新症状提取。"""
    items_lines = []
    for q in questions:
        if q.get("type") == "slot":
            items_lines.append(f"  - 补全 HPI 维度 {q['slot']}(回填到 slot_fills)")
        elif q.get("type") == "open":
            items_lines.append("  - 开放式问『还有别的不舒服』(新症状回填到 new_symptoms)")
    items_block = "\n".join(items_lines) if items_lines else "  (无)"

    return f"""你是问诊回答解析助手。请把患者回答结构化。

【追问问题】
{followup_question}

【本轮追问的待回答项】
{items_block}

【患者回答】
{followup_answer}

【解析规则】
1. 维度级回填(slot_fills,key=槽位名):
   - 单值槽(onset_time/onset_mode/trigger/location/nature/severity/
     duration_pattern/progression/treatment_tried/treatment_response):value=str
   - 多值槽(aggravating/relieving/associated_symptoms):value=list[str]
   - 槽位名必须是 HPI 13 维之一,不要新造槽名;患者没涉及的槽位**不要**出现在 slot_fills 里
2. new_symptoms:患者回答里**主动提到的症状**(无论是开放式问的回答,还是顺带补充);
   - 用患者原文或常见医学短语,不要太长(如"反酸"、"右上腹放射痛"、"夜间盗汗")
   - 若回答只涉及维度填补、未提及任何新症状,则为空列表
   - 已确认 / 已否认 / 不确定列表中的术语**不要**重复输出""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ⑧a recommend_exam(自由文本输出)
# ────────────────────────────────────────────────────────────────────────────


def build_recommend_exam_prompt(
    diagnosis_results: list[dict],
    unaskable_symptoms: list[dict],
    candidate_chunks_preview: list[str],
    existing_report_findings: list[dict],
) -> str:
    """⑧a recommend_exam(自由文本):基于诊断结果 + 不可问体征推断需要的检查。

    `unaskable_symptoms` 是 ⑩ Step 3 精筛过的版本(`{description, reason}` 结构),
    可直接据 description 拟检查建议。
    """
    diag_lines = [
        f"  - {r.get('disease')} (p={r.get('probability', 0):.2f}, type={r.get('differentiation_type')})"
        for r in diagnosis_results[:5]
    ]
    diag_block = "\n".join(diag_lines) or "  (无诊断结果)"

    unaskable_lines = [
        f"  - {u.get('description')} —— {u.get('reason')}"
        for u in unaskable_symptoms[:8]
    ]
    unaskable_block = "\n".join(unaskable_lines) or "  (无)"

    chunks_preview = "\n".join(f"  - {c[:80]}" for c in candidate_chunks_preview[:3]) or "  (无)"

    existing_lines = []
    for r in existing_report_findings[:5]:
        existing_lines.append(
            f"  - {r.get('report_type')} ({r.get('report_date')}): "
            f"impressions={r.get('impressions')[:2]}"
        )
    existing_block = "\n".join(existing_lines) or "  (无)"

    return f"""你是医生检查建议助手。请基于诊断候选 + 鉴别要点,推荐 3-5 项检查,按优先级排序,
**不要静默删除已有报告对应的检查**——对已有报告的项,额外加复用评估说明。

【诊断候选】
{diag_block}

【需检查鉴别的体征(unaskable)】
{unaskable_block}

【相关文献片段(供参考)】
{chunks_preview}

【患者已有报告】
{existing_block}

【输出格式】
按优先级编号列出检查,每项 1-2 句说明:
1. 检查名(优先级原因)
2. ...

对与已有报告交集的检查,在该项里追加"已有[日期]报告,可携带评估是否需要复做"
之类的复用说明,不要直接删掉。

口吻面向患者,避免医学术语堆砌,涉及禁食/造影剂等特殊条件要写明。""" + _JSON_TAIL


# ────────────────────────────────────────────────────────────────────────────
# ⑩ diagnose 1 步 prompt(对齐评测口径 .eval/rag_eval/run_diagnose_eval.py)
# ────────────────────────────────────────────────────────────────────────────


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
    history_summary: str,
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
            history_summary / report_findings: 多轮 followup 累积的患者画像
        unaskable_symptoms: ⑤ 写入的粗筛版({description, reason}),供 LLM 产 retained_unaskable
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

【多轮交互累积的患者画像】
- 已确认症状:{confirmed_block}
- 已否认症状:{denied_block}
- 已问但不确定的症状:{uncertain_block}

【病史摘要】
{history_summary or "(无)"}

【检查报告发现】
{reports_block}

【⑤ 写入的 unaskable 粗筛(LLM 想知道但患者答不上的体征,供 retained_unaskable 精筛参考)】
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
3. retained_unaskable(基于诊断结果挑/改写,从【⑤ 写入的 unaskable 粗筛】里精筛):
   - top1=`confirmed` → 通常返空列表(证据已闭环,无需再查)
   - top1=`insufficient` → 通常返空列表(检查也救不回信息不足)
   - top1=`need_exam` → **至少保留 1 条**,只留对当前 top 候选鉴别真正关键的;描述可改写
     得更聚焦,如把"想知道胆囊有无问题"改成"腹部 B 超确认胆囊壁厚度 + 有无结石"
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
