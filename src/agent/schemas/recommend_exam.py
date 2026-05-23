"""Agent ⑧a recommend_exam LLM 输出 schema(DEV_SPEC §9.5)。

2026-05-22 改造:从 `tests: list[str]` → `test_groups: list[ExamGroup]`,
按"医院实际出报告载体"分组(化验合 1 组,体格合 1 组,影像各自独立,...),
UI 据此给每组出独立上传框,⑨ 解析时拿 group_label 作 hint 提升准确度。

`rationale` 是 LLM 整体说明(为什么推荐这些 / 已有哪些可复用),作为审计 + prompt
透传给 ⑫。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExamGroup(BaseModel):
    """一组共享同一份报告载体的检查项(供 UI 渲染独立上传框)。"""

    group_label: str = Field(
        ...,
        description=(
            "给患者看的组名,如 '抽血化验(空腹8h)' / '医生体格检查记录' / "
            "'腹部 B 超' / '心电图'。简洁中文,不堆叠多个项目名。"
        ),
    )
    items: list[str] = Field(
        default_factory=list,
        description=(
            "该组包含的具体检查项 list[str],如 ['血常规', '肝功能', '淀粉酶', '脂肪酶']。"
            "单项也用 list 表达(如 ['腹部 B 超(肝胆胰脾)'])。最多 6 条避免组名/UI 拥挤。"
        ),
    )
    note: str = Field(
        default="",
        description=(
            "给患者的简短解释,为什么这些放在一起 / 怎么准备 / 拍什么。"
            "例如 '医院一次抽血出多个指标,一张化验单' / "
            "'拍门诊病历本对应页即可' / '需空腹 6-8 小时'。1-2 句。"
        ),
    )


class RecommendExamOutput(BaseModel):
    """⑧a recommend_exam LLM 结构化输出。"""

    test_groups: list[ExamGroup] = Field(
        default_factory=list,
        description=(
            "建议的检查项,按报告载体分组(2026-05-22 从扁平 tests list[str] 改造)。"
            "典型 2-5 组,覆盖 4 大类:"
            "  - 抽血化验类(合 1 组,医院一次抽血)"
            "  - 体格检查类(合 1 组,医生当场判断写病历)"
            "  - 影像类(各自独立,每项独立报告)"
            "  - 心电图/病理/内镜(各自独立)。"
            "若所有所需检查患者都已上传报告则可为空。"
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "整体推荐理由(为什么这几项 / 已有哪些可复用 / 哪个对鉴别最关键);"
            "面向患者的简短说明,不是逐项理由,2-3 句即可"
        ),
    )
