"""tests/unit/test_node_exam_loop.py — F9 检查循环单元测试。

⑧a recommend_exam(2026-05-21 起**双模式**:首诊 vs 鉴别,看 diagnosis_result 是否非空切;
不再读 candidate_chunks)+ ⑧b wait_exam_report(interrupt)+ ⑨ process_exam_result
(复用 parse_reports)。
"""
from __future__ import annotations

from unittest.mock import patch

from src.agent.schemas.recommend_exam import RecommendExamOutput
from src.agent.state import create_initial_state


# ⑧a recommend_exam — 鉴别模式(diagnosis_result 非空, ⑩ 后 need_exam 触发)


@patch("src.agent.nodes.recommend_exam.build_recommend_exam_prompt")
@patch("src.agent.nodes.recommend_exam.get_llm")
def test_recommend_exam_differential_mode_when_diagnosis_present(mock_llm_factory, mock_build):
    from src.agent.nodes.recommend_exam import recommend_exam

    mock_chain = (
        mock_llm_factory.return_value
        .with_structured_output.return_value
        .with_retry.return_value
    )
    # 2026-05-22:输出从 tests: list[str] 改为 test_groups: list[ExamGroup]
    from src.agent.schemas.recommend_exam import ExamGroup
    mock_chain.invoke.return_value = RecommendExamOutput(
        test_groups=[
            ExamGroup(group_label="腹部 B 超", items=["腹部超声"], note=""),
            ExamGroup(group_label="胃镜", items=["胃镜"], note=""),
            ExamGroup(group_label="腹部 B 超", items=["腹部超声"], note=""),  # 重名 → 去重
        ],
        rationale="腹部超声优先,可区分胆囊炎;胃镜可确认溃疡",
    )
    mock_build.return_value = "mocked-prompt"

    s = create_initial_state(patient_id="P", patient_input="x")
    s.diagnosis_result = [
        {"disease": "胆囊炎", "probability": 0.5, "differentiation_type": "need_exam"}
    ]
    update = recommend_exam(s)
    assert update["exam_round"] == 1
    # spec §9.5:recommended_test_groups list[dict],group_label 去重
    groups = update["recommended_test_groups"]
    assert [g["group_label"] for g in groups] == ["腹部 B 超", "胃镜"]
    assert groups[0]["items"] == ["腹部超声"]
    # 验证 prompt 走鉴别模式
    assert mock_build.call_args.kwargs["mode"] == "differential"


@patch("src.agent.nodes.recommend_exam.build_recommend_exam_prompt")
@patch("src.agent.nodes.recommend_exam.get_llm")
def test_recommend_exam_intake_mode_when_no_diagnosis(mock_llm_factory, mock_build):
    """⑤ 触发的首诊模式:无 diagnosis_result,基于主诉 + ⑤ 写的 unaskable_symptoms 推首诊全套。"""
    from src.agent.nodes.recommend_exam import recommend_exam

    mock_chain = (
        mock_llm_factory.return_value
        .with_structured_output.return_value
        .with_retry.return_value
    )
    # 2026-05-22:分组输出(抽血化验合一组;影像独立)
    from src.agent.schemas.recommend_exam import ExamGroup
    mock_chain.invoke.return_value = RecommendExamOutput(
        test_groups=[
            ExamGroup(
                group_label="抽血化验(空腹8h)",
                items=["血常规", "肝功能"],
                note="一次抽血出多项",
            ),
            ExamGroup(group_label="腹部 B 超", items=["腹部 B 超"], note="需空腹 6-8h"),
        ],
        rationale="腹痛首诊标准三件套",
    )
    mock_build.return_value = "mocked-prompt"

    s = create_initial_state(patient_id="P", patient_input="腹痛 3 天")
    s.chief_complaint = "腹痛"
    s.present_illness = "右上腹痛 3 天,饭后加重"
    s.diagnosis_result = []  # 空 → 首诊模式
    s.unaskable_symptoms = [
        {"description": "Murphy 征查体", "reason": "鉴别胆囊炎 vs 胃炎"},
        {"description": "腹部 B 超", "reason": "排查胆石"},
    ]

    update = recommend_exam(s)
    assert update["exam_round"] == 1
    groups = update["recommended_test_groups"]
    assert [g["group_label"] for g in groups] == ["抽血化验(空腹8h)", "腹部 B 超"]
    assert groups[0]["items"] == ["血常规", "肝功能"]
    assert groups[0]["note"] == "一次抽血出多项"
    # 验证 prompt 走首诊模式 + 透传 unaskable_symptoms
    assert mock_build.call_args.kwargs["mode"] == "intake"
    assert mock_build.call_args.kwargs["unaskable_symptoms"] == [
        {"description": "Murphy 征查体", "reason": "鉴别胆囊炎 vs 胃炎"},
        {"description": "腹部 B 超", "reason": "排查胆石"},
    ]


@patch("src.agent.nodes.recommend_exam.get_llm")
def test_recommend_exam_empty_tests_yields_empty_list(mock_llm_factory):
    """所有所需检查患者已上传报告 → tests 可为空(spec §9.5 RecommendExamOutput)。"""
    from src.agent.nodes.recommend_exam import recommend_exam

    mock_chain = (
        mock_llm_factory.return_value
        .with_structured_output.return_value
        .with_retry.return_value
    )
    mock_chain.invoke.return_value = RecommendExamOutput(test_groups=[], rationale="已有报告全覆盖")

    s = create_initial_state(patient_id="P", patient_input="x")
    update = recommend_exam(s)
    assert update["recommended_test_groups"] == []
    assert update["exam_round"] == 1


# ⑧b wait_exam_report


@patch("src.agent.nodes.wait_exam_report.interrupt")
def test_wait_exam_report_calls_interrupt(mock_interrupt):
    from src.agent.nodes.wait_exam_report import wait_exam_report

    # 2026-05-22:interrupt payload 改成 recommended_test_groups(分组结构)
    mock_interrupt.return_value = [
        {"group_label": "腹部 B 超", "files": ["/tmp/lab.pdf"], "status": "uploaded"},
    ]
    s = create_initial_state(patient_id="P", patient_input="x")
    s.recommended_test_groups = [
        {"group_label": "腹部 B 超", "items": ["腹部超声"], "note": ""},
    ]
    update = wait_exam_report(s)
    mock_interrupt.assert_called_once_with(
        [{"group_label": "腹部 B 超", "items": ["腹部超声"], "note": ""}]
    )
    assert update == {
        "pending_exam_results": [
            {"group_label": "腹部 B 超", "files": ["/tmp/lab.pdf"], "status": "uploaded"},
        ]
    }


# ⑨ process_exam_result


@patch("src.agent.nodes.process_exam_result.parse_reports")
def test_process_exam_result_appends_reports_and_findings(mock_parse):
    from src.agent.nodes.process_exam_result import process_exam_result

    mock_parse.return_value = [
        {
            "report_type": "imaging",
            "report_date": "2026-05-12",
            "report_index": 0,
            "abnormal_values": [],
            "impressions": ["胆囊炎征象"],
            "positive_findings": ["胆囊壁增厚"],
            "negative_findings": [],
        }
    ]

    s = create_initial_state(patient_id="P", patient_input="x")
    s.exam_reports = [{"file_ref": "/already/old.jpg"}]  # base = 1
    s.pending_exam_results = [{"file_ref": "/new/lab.pdf"}]
    update = process_exam_result(s)

    assert len(update["exam_reports"]) == 2
    assert update["report_findings"][0]["report_index"] == 1  # base + 0
    assert update["pending_exam_results"] == []


def test_process_exam_result_empty_pending_returns_empty():
    from src.agent.nodes.process_exam_result import process_exam_result

    s = create_initial_state(patient_id="P", patient_input="x")
    update = process_exam_result(s)
    assert update == {}
