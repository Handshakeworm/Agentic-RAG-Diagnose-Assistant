"""src/agent/nodes/initial_ask.py — Agent ⓪a initial_ask 模板追问(DEV_SPEC §4.1.3 流程图)。

职责:**一次性交互节点**(零 LLM,纯模板)
- 加载患者 basic_info.gender(决定要不要问妊娠/哺乳)
- 拼装首轮模板问题写入 state.followup_questions:
    1. open    : "您还有其他不适吗?"
    2. history : "您有什么过敏/慢病/长期用药?"(从 ④ 鉴别诊断追问中剥离的病史采集)
    3. obstetric (仅 gender == 'female'):"您当前是否怀孕/哺乳?"
       (是当下状态,以本会话回答为准,不看记录新鲜度)
- 拼 followup_question 文本(用 intake_followup_ask 同款模板 helper)
- **节点入口 `interrupt(...)` 暂停,等用户答完综合 form 才推进**;resume 后写
  followup_answer,交给下游 ⑦ process_followup_answer LLM 解析(history → medical_history
  merge,obstetric → obstetric_history merge,open 新症状 → confirmed_symptoms)。

为什么 ⓪a 必须自己 interrupt:用户体感上"提交症状 → 第一个 form 是 open/病史/孕期"
是这个节点存在的全部意义。把"准备 form 内容"和"弹给用户"拆到两个节点(原来交给 ⑥)
会让 ①.5 的报告 form 跑在 ⓪a form 之前 —— 跟节点编号顺序不一致,体验混乱。

模型:零 LLM(LLM 解析归 ⑦)。
"""
from __future__ import annotations

from langgraph.types import interrupt

from src.agent.nodes.intake_followup_ask import (
    _attach_question_texts,
    _build_question_text,
)
from src.agent.state import MedicalState
from src.agent.utils.patient_repo import load_medical_history


def initial_ask(state: MedicalState) -> dict:
    """模板组装 + interrupt 等用户答;resume 后把 answer 写到 followup_answer 给 ⑦ 解析。

    Returns:
      首次执行(raise GraphInterrupt 前):无返回(被 LangGraph 拦)
      resume 后再跑(interrupt 直接返回 user answer):
        {"followup_questions": ..., "followup_question": ..., "followup_answer": <user>}
    """
    profile = load_medical_history(state.patient_id)
    gender = (profile.get("basic_info") or {}).get("gender")

    questions: list[dict] = [
        {"type": "open"},
        {"type": "history"},
    ]
    if gender == "female":
        questions.append({"type": "obstetric"})

    # 给每条 dict 补 question 字段(供前端 form 渲染) + 拼整段文案(后端 SSE / ⑦ 解析共用)
    questions = _attach_question_texts(questions)
    question_text = _build_question_text(questions)

    # 节点入口 interrupt:第一次跑 raise GraphInterrupt(SSE 推 ongoing_initial_ask
    # form);client 提交答案后 resume,interrupt(...) 直接返回 user answer 字符串
    user_answer = interrupt({
        "followup_questions": questions,
        "followup_question": question_text,
    })

    return {
        "followup_questions": questions,
        "followup_question": question_text,
        "followup_answer": user_answer if isinstance(user_answer, str) else "",
        "medical_history": profile,  # 既然已经 load,顺手存进去,后续不用重 load
    }
