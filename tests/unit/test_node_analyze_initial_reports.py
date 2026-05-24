"""tests/unit/test_node_analyze_initial_reports.py — F2.5 ①.5 单元测试。

验证 spec §4.1.2 ①.5(2026-05-23 重构后两条路径):

**上传路径**(interrupt resume value = list[group dict]):
- 单组 uploaded → parse_reports 被调,hint 含 group_label,exam_reports 各项带 group_label
- 多组含 skipped → 只解析 status='uploaded' 且 files 非空的组
- 上传路径**不**走 PG fallback(用户主动传新报告就不混老的)

**跳过路径**(interrupt resume value = 空 list / 空 str / 其他兼容降级):
- exam_reports PG 为空 → early return 透传,不发起 LLM
- exam_reports PG 非空 → load_initial_exam_reports + parse_reports
- LLM 失败 → 降级为空 findings(不阻塞流水线)

①.5 节点入口有 `interrupt(...)` 问"要不要上传报告"。unit 直接调函数不在 LangGraph
runnable context,interrupt() 会抛 RuntimeError。各 test 用 `@patch` 显式控制 resume 值。
"""
from __future__ import annotations

from unittest.mock import patch

from src.agent.schemas.report_parser import ReportFinding, ReportFindings
from src.agent.state import create_initial_state


# ─── 跳过路径(resume = 空 / 字符串降级 → PG fallback) ───────────────────


@patch("src.agent.nodes.analyze_initial_reports.interrupt", return_value=[])
@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports", return_value=[])
def test_skip_empty_pg_early_returns_no_llm_call(_load, _interrupt):
    """跳过 + PG 空 → 只写 exam_reports=[] 透传,不调 parse_reports。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    s = create_initial_state(patient_id="P", patient_input="x")
    with patch("src.agent.nodes.analyze_initial_reports.parse_reports") as mock_parse:
        update = analyze_initial_reports(s)
    assert update == {"exam_reports": []}
    mock_parse.assert_not_called()


@patch("src.agent.nodes.analyze_initial_reports.interrupt", return_value="")
@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports",
       return_value=[{"file_ref": "/tmp/r1.jpg"}, {"file_ref": "/tmp/r2.pdf"}])
@patch("src.agent.utils.report_parser.get_llm")
@patch(
    "src.agent.utils.report_parser._build_multimodal_message",
    return_value="<msg-stub>",
)
def test_skip_with_pg_history_parses_reports(_msg, mock_llm_factory, _load, _interrupt):
    """跳过 + PG 有历史报告 → load_initial_exam_reports + LLM 返 2 个 finding,补 report_index。"""
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

    s = create_initial_state(patient_id="P-TEST", patient_input="腹痛")
    update = analyze_initial_reports(s)

    findings = update["report_findings"]
    assert len(findings) == 2
    assert findings[0]["report_index"] == 0
    assert findings[1]["report_index"] == 1
    assert findings[0]["report_type"] == "blood_routine"
    assert findings[1]["impressions"] == ["右上腹胆囊壁增厚"]


@patch("src.agent.nodes.analyze_initial_reports.interrupt", return_value="")
@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports",
       return_value=[{"file_ref": "/tmp/r.jpg"}])
@patch("src.agent.utils.report_parser.get_llm")
@patch(
    "src.agent.utils.report_parser._build_multimodal_message",
    return_value="<msg-stub>",
)
def test_skip_llm_failure_returns_empty_findings_does_not_raise(_msg, mock_llm_factory, _load, _interrupt):
    """跳过 + PG 有历史 + LLM 失败 → 降级空 findings,不抛(spec §9.3 中级失败处理)。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    mock_chain = mock_llm_factory.return_value.with_structured_output.return_value.with_retry.return_value
    mock_chain.invoke.side_effect = RuntimeError("multimodal LLM rejected request")

    s = create_initial_state(patient_id="P", patient_input="腹痛")
    update = analyze_initial_reports(s)
    assert update == {"exam_reports": [{"file_ref": "/tmp/r.jpg"}], "report_findings": []}


# ─── 上传路径(resume = list[group dict] → 真调 parse_reports + hint) ────


@patch("src.agent.nodes.analyze_initial_reports.parse_reports_parallel")
@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports")
@patch("src.agent.nodes.analyze_initial_reports.interrupt")
def test_upload_single_group_calls_parse_with_hint(mock_interrupt, mock_load, mock_parallel):
    """单组 uploaded → parse_reports_parallel 单 task 调一次,hint 含 group_label,**不**走 PG fallback。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    mock_interrupt.return_value = [
        {"group_label": "血常规", "files": ["/uploads/blood1.jpg", "/uploads/blood2.jpg"], "status": "uploaded"},
    ]
    mock_parallel.return_value = [[
        {
            "report_type": "blood_routine",
            "report_date": "2026-05-10",
            "report_index": 0,
            "abnormal_values": ["WBC 12.3↑"],
            "impressions": [],
            "positive_findings": ["白细胞升高"],
            "negative_findings": [],
        },
    ]]

    s = create_initial_state(patient_id="P", patient_input="腹痛")
    update = analyze_initial_reports(s)

    # PG fallback 不该被触发
    mock_load.assert_not_called()
    # parse_reports_parallel 调一次,tasks=[(files, hint)] 单 task
    mock_parallel.assert_called_once()
    tasks = mock_parallel.call_args.args[0]
    assert len(tasks) == 1
    files, hint = tasks[0]
    assert files == ["/uploads/blood1.jpg", "/uploads/blood2.jpg"]
    assert "血常规" in (hint or "")

    # exam_reports 每项带 group_label
    assert update["exam_reports"] == [
        {"file_ref": "/uploads/blood1.jpg", "group_label": "血常规"},
        {"file_ref": "/uploads/blood2.jpg", "group_label": "血常规"},
    ]
    # report_findings 带 group_label + report_index 从 0 起
    assert update["report_findings"][0]["group_label"] == "血常规"
    assert update["report_findings"][0]["report_index"] == 0


@patch("src.agent.nodes.analyze_initial_reports.parse_reports_parallel")
@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports")
@patch("src.agent.nodes.analyze_initial_reports.interrupt")
def test_upload_multiple_groups_some_skipped(mock_interrupt, mock_load, mock_parallel):
    """多组上传,其中一组 status=skipped → 并行 parse 只跑非 skipped 的组,report_index 跨组连续。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    mock_interrupt.return_value = [
        {"group_label": "血常规", "files": ["/uploads/blood.jpg"], "status": "uploaded"},
        {"group_label": "B超", "files": [], "status": "skipped"},  # skipped 跳过
        {"group_label": "心电图", "files": ["/uploads/ecg.pdf"], "status": "uploaded"},
    ]
    # parse_reports_parallel 返回与 tasks 等长的 list[list[finding]](顺序同 tasks)
    mock_parallel.return_value = [
        [{"report_type": "blood_routine", "report_index": 0, "impressions": ["白细胞升高"]}],
        [{"report_type": "ecg", "report_index": 0, "impressions": ["窦性心律"]}],
    ]

    s = create_initial_state(patient_id="P", patient_input="腹痛")
    update = analyze_initial_reports(s)

    mock_load.assert_not_called()
    # 只有 2 个 task(B超 skipped 不进 tasks)
    tasks = mock_parallel.call_args.args[0]
    assert len(tasks) == 2

    # exam_reports 只有 2 项(血常规 + 心电图,B超 跳过)
    assert len(update["exam_reports"]) == 2
    assert update["exam_reports"][0]["group_label"] == "血常规"
    assert update["exam_reports"][1]["group_label"] == "心电图"

    # report_findings 跨组连续:0, 1
    findings = update["report_findings"]
    assert len(findings) == 2
    assert findings[0]["report_index"] == 0
    assert findings[0]["group_label"] == "血常规"
    assert findings[1]["report_index"] == 1
    assert findings[1]["group_label"] == "心电图"


@patch("src.agent.nodes.analyze_initial_reports.parse_reports_parallel")
@patch("src.agent.nodes.analyze_initial_reports.load_initial_exam_reports")
@patch("src.agent.nodes.analyze_initial_reports.interrupt")
def test_upload_group_without_label_still_parses(mock_interrupt, mock_load, mock_parallel):
    """group_label 为空(患者没填标签)→ 仍解析,hint=None,exam_reports 不带 group_label。"""
    from src.agent.nodes.analyze_initial_reports import analyze_initial_reports

    mock_interrupt.return_value = [
        {"group_label": "", "files": ["/uploads/unknown.pdf"], "status": "uploaded"},
    ]
    mock_parallel.return_value = [[
        {"report_type": "other", "report_index": 0, "impressions": ["未知"]},
    ]]

    s = create_initial_state(patient_id="P", patient_input="腹痛")
    update = analyze_initial_reports(s)

    mock_load.assert_not_called()
    # hint 应为 None(label 空)
    tasks = mock_parallel.call_args.args[0]
    _files, hint = tasks[0]
    assert hint is None
    # exam_reports 不带 group_label key
    assert update["exam_reports"] == [{"file_ref": "/uploads/unknown.pdf"}]
    # finding 也不带 group_label
    assert "group_label" not in update["report_findings"][0]
