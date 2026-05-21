"""src/agent/nodes/safety_gate.py — Agent ⑪ safety_gate(DEV_SPEC §4.1.2 ⑪)。

两层结构(spec §4.1.2 ⑪ "病史信息分层接入机制 — 安全门控层"):

  规则过滤(确定性):从结构化 medical_history 抽 allergy / medication /
                    pregnancy 状态,执行确定性匹配
                    → banned_drugs / interaction_warnings / contraindication_flags

  LLM 兜底(高安全级):规则层覆盖不到的(交叉过敏 / 罕见相互作用 / 肝肾剂量调整)
                     → SafetyGateOutput.additional_risks 追加到约束

**重构标记**(spec §4.1.2 ⑪ TODO):当前规则层只做"病史结构化字段 → 简单匹配",
还没有把《临床用药指南》解析进 PostgreSQL 规则表(drug_allergy_rules /
drug_interaction_rules / drug_pregnancy_categories)。本实现:
- 规则层 = 把 medical_history 中的过敏药和当前用药直接列出来,标记为 banned/
  interaction(简单"禁用列表"风格,而不是 RAG 检索)
- 规则知识库未建,所以暂未做交叉/相互作用判断,这部分完全交给 LLM 兜底
- 等用药指南规则表建好后,在这里加 SQL 查询,LLM 兜底比例自然下降

LLM 失败 → 保守路径:additional_risks 追加 "LLM 安全评估不可用,建议线下由药师复核"
通用警告(spec §9.3 高安全等级失败处理)。
"""
from __future__ import annotations

import logging
import time

from src.agent.schemas.safety_gate import SafetyGateOutput, SafetyRisk
from src.agent.state import MedicalState
from src.common.metrics import (
    _attempts,
    _failures,
    _fallbacks,
    _latency,
    retry_observer,
)
from src.models.llm_client import get_llm
from src.prompts.agent import build_safety_gate_prompt


_logger = logging.getLogger(__name__)
_NODE = "safety_gate_llm"
_SCHEMA = "SafetyGateOutput"

_LLM_UNAVAILABLE_MSG = "LLM 安全评估不可用,建议线下由药师复核"


def _rule_layer_constraints(medical_history: dict) -> dict:
    """规则层确定性约束(直接从结构化字段提取,不走 RAG)。

    Returns:
        {
          "banned_drugs":           list[str],  # 过敏药名 + 同名同类(规则简单版)
          "interaction_warnings":   list[dict], # 当前用药 → 与候选用药相互作用占位(暂空)
          "contraindication_flags": dict,       # pregnancy / lactation 等
        }
    """
    banned: list[str] = []
    for allergy in medical_history.get("allergy_history") or []:
        if isinstance(allergy, dict):
            name = allergy.get("substance") or allergy.get("drug") or allergy.get("name")
        else:
            name = str(allergy)
        if name and name not in banned:
            banned.append(name)

    interaction_warnings: list[dict] = []  # 规则知识库待建,暂空(交给 LLM 兜底)

    flags: dict = {}
    obstetric = medical_history.get("obstetric_history") or {}
    if isinstance(obstetric, dict):
        if obstetric.get("pregnancy_status") in ("pregnant", "Pregnant", True):
            flags["pregnancy"] = True
        if obstetric.get("lactation_status") in ("lactating", True):
            flags["lactation"] = True

    return {
        "banned_drugs": banned,
        "interaction_warnings": interaction_warnings,
        "contraindication_flags": flags,
    }


def _call_llm_fallback(
    diagnosis_result: list[dict],
    medical_history: dict,
    rule_constraints: dict,
) -> SafetyGateOutput:
    """LLM 兜底(高安全级)— 失败保守路径:additional_risks 追加通用警告。"""
    prompt = build_safety_gate_prompt(
        diagnosis_results=diagnosis_result,
        medical_history=medical_history,
        rule_layer_constraints=rule_constraints,
    )

    _attempts.labels(node=_NODE, schema=_SCHEMA).inc()
    t0 = time.perf_counter()
    try:
        chain = get_llm().with_structured_output(SafetyGateOutput, method="json_mode").with_retry(stop_after_attempt=3)
        return chain.invoke(
            prompt,
            config={
                "callbacks": [retry_observer],
                "metadata": {"node": _NODE, "schema": _SCHEMA},
            },
        )
    except Exception as e:
        _failures.labels(
            node=_NODE, schema=_SCHEMA, exception_type=type(e).__name__
        ).inc()
        _fallbacks.labels(node=_NODE, fallback_type="safety_conservative").inc()
        _logger.error("[%s] LLM fallback failed, conservative warning: %s", _NODE, e)
        return SafetyGateOutput(
            additional_risks=[
                SafetyRisk(
                    risk_type="cross_allergy",  # 占位类型,description 已说明语义
                    description=_LLM_UNAVAILABLE_MSG,
                    severity="medium",
                    recommendation="线下就诊,由执业药师复核用药安全",
                )
            ]
        )
    finally:
        elapsed = time.perf_counter() - t0
        _latency.labels(node=_NODE, schema=_SCHEMA).observe(elapsed)
        _logger.info("[%s] elapsed=%.2fs", _NODE, elapsed)


def safety_gate(state: MedicalState) -> dict:
    rule = _rule_layer_constraints(state.medical_history)
    llm_result = _call_llm_fallback(state.diagnosis_result, state.medical_history, rule)

    return {
        "safety_constraints": {
            "banned_drugs": rule["banned_drugs"],
            "interaction_warnings": rule["interaction_warnings"],
            "contraindication_flags": rule["contraindication_flags"],
            "additional_risks": [r.model_dump() for r in llm_result.additional_risks],
        }
    }
