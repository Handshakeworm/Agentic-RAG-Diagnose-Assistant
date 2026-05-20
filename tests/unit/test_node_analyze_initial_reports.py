"""tests/unit/test_node_analyze_initial_reports.py — F2.5 ①.5 单元测试。

验证 spec §4.1.2 ①.5:
- exam_reports 为空 → early return 透传(返回空 dict,不发起 LLM)
- exam_reports 非空 → 调 parse_reports + report_index 自动补
- LLM 失败 → 降级为空 findings(不阻塞流水线)

①.5 节点入口有 `interrupt(...)` 问"要不要上传报告"。unit 直接调函数不在 LangGraph
runnable context,interrupt() 会抛 RuntimeError。autouse fixture 把它 patch 成默认
返空串(= "用户跳过"),保持原 test 路径不变;真 interrupt 行为由集成测覆盖。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.schemas.report_parser import ReportFinding, ReportFindings
from src.agent.state import create_initial_state


@pytest.fixture(autouse=True)
def _patch_interrupt():
    with patch(
        "src.agent.nodes.analyze_initial_reports.interrupt", return_value=""
    ):
        yield


def _state_with_reports(reports: list[dict]):
    s = create_initial_state(patient_id="P-TEST", patient_input="腹痛")
    s.exam_reports = reports
    return s


@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports", return_value=[])
def test_empty_reports_early_returns_no_llm_call(_load):
    """exam_reports 为空 → 只写 exam_reports=[] 透传,不调 parse_reports。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    s = create_initial_state(patient_id="P", patient_input="x")
    with patch("src.agent.nodes.analyze_initial_reports.parse_reports") as mock_parse:
        update = analyze_initial_reports(s)
    assert update == {"exam_reports": []}
    mock_parse.assert_not_called()


@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports",
       return_value=[{"file_ref": "/tmp/r1.jpg"}, {"file_ref": "/tmp/r2.pdf"}])
@patch("src.agent.utils.report_parser.get_llm")
@patch(
    "src.agent.utils.report_parser._build_multimodal_message",
    return_value="<msg-stub>",
)
def test_two_reports_parsed_with_report_index(_msg, mock_llm_factory, _load):
    """两份报告 → LLM 返回 2 个 finding,自动补 report_index 0/1。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.return_value = ReportFindings(
        findings=[
            ReportFinding(
                report_type="blood_routine",
                report_date="2026-05-01",
                abnormal_values=["WBC 12.3↑"],
                impressions=[],
                positive_findings=["白细胞升高"],
                negative_findings=[],
            ),
            ReportFinding(
                report_type="imaging",
                report_date="2026-05-03",
                abnormal_values=[],
                impressions=["右上腹胆囊壁增厚"],
                positive_findings=["胆囊炎征象"],
                negative_findings=["未见胆管扩张"],
            ),
        ]
    )

    state = _state_with_reports(
        [{"file_ref": "/tmp/r1.jpg"}, {"file_ref": "/tmp/r2.pdf"}]
    )
    update = analyze_initial_reports(state)

    findings = update["report_findings"]
    assert len(findings) == 2
    assert findings[0]["report_index"] == 0
    assert findings[1]["report_index"] == 1
    assert findings[0]["report_type"] == "blood_routine"
    assert findings[1]["impressions"] == ["右上腹胆囊壁增厚"]


@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports",
       return_value=[{"file_ref": "/tmp/r.jpg"}])
@patch("src.agent.utils.report_parser.get_llm")
@patch(
    "src.agent.utils.report_parser._build_multimodal_message",
    return_value="<msg-stub>",
)
def test_llm_failure_returns_empty_findings_does_not_raise(_msg, mock_llm_factory, _load):
    """LLM 失败 → 降级返回空 findings,不抛异常(spec §9.3 中级失败处理)。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = RuntimeError("multimodal LLM rejected request")

    state = _state_with_reports([{"file_ref": "/tmp/r.jpg"}])
    update = analyze_initial_reports(state)
    assert update == {"exam_reports": [{"file_ref": "/tmp/r.jpg"}], "report_findings": []}
