"""tests/unit/test_settings.py — 锁住 DEV_SPEC §9.7 七常量初始值 + env 覆盖能力。

这些断言不是"测代码逻辑",是把 spec 文本契约钉死在测试里:
任何后续修改若破坏 §9.7.1 表的初始值,或破坏 §9.7.4 "改 .env 不改代码"的覆盖能力,
本文件会 fail,提醒 reviewer 同步去看 spec。
"""

from __future__ import annotations

import importlib

import pytest


# ────────────────────────────────────────────────────────────────────────────
# §9.7.1 初始值锁定
# ────────────────────────────────────────────────────────────────────────────


def test_agent_limits_initial_values_match_spec_9_7_1() -> None:
    """DEV_SPEC §9.7.1 常量清单的 7 个初始值必须与 spec 一字不差。

    锁的是 Pydantic Field 的 **class default**(spec 文本契约),不是 settings
    实例的 effective value(后者会被 .env / 进程 env 覆盖,见 §9.7.4)。
    """
    from config.settings import AgentLimitsSettings

    fields = AgentLimitsSettings.model_fields
    assert fields["MAX_FOLLOWUP_ROUNDS"].default == 8
    assert fields["MAX_EXAM_ROUNDS"].default == 3
    assert fields["MAX_FOLLOWUP_QUESTIONS"].default == 5
    assert fields["RETRIEVE_TOP_N"].default == 200
    # 注:ENTITY_LINKING_TIER2_THRESHOLD 字段已随 EL 移除一并删除
    # ASKABLE_GAIN_THRESHOLD 已随 ⑤ 重设计删除(信息增益机制不再使用)
    assert "ENTITY_LINKING_TIER2_THRESHOLD" not in fields
    assert "ASKABLE_GAIN_THRESHOLD" not in fields
    assert fields["RERANKER_CUTOFF_LAYERS"].default is None  # 全 40 层(模型 layerwise 完整深度)


# ────────────────────────────────────────────────────────────────────────────
# §9.7.4 "改 .env 不改代码" 覆盖能力
# ────────────────────────────────────────────────────────────────────────────


def test_env_var_overrides_agent_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT_MAX_FOLLOWUP_ROUNDS=10 必须能在不改代码的前提下覆盖默认 8。"""
    monkeypatch.setenv("AGENT_MAX_FOLLOWUP_ROUNDS", "10")
    monkeypatch.setenv("AGENT_MAX_EXAM_ROUNDS", "5")

    import config.settings as mod

    importlib.reload(mod)  # 重读 env
    try:
        assert mod.settings.agent_limits.MAX_FOLLOWUP_ROUNDS == 10
        assert mod.settings.agent_limits.MAX_EXAM_ROUNDS == 5
    finally:
        # reload 恢复原始环境(monkeypatch 退出时会自动 unset env,这里再 reload 一次让单例回到默认)
        monkeypatch.undo()
        importlib.reload(mod)


# ────────────────────────────────────────────────────────────────────────────
# 顶层段聚合 + DSN 拼接
# ────────────────────────────────────────────────────────────────────────────


def test_all_settings_sections_accessible() -> None:
    """settings.<段>.<字段> 全部段必须能访问,防止段被误删/改名。"""
    from config.settings import settings

    # 11 个段(10 个子段 + 1 个顶层 ENV);chunking 段已删除(2026-05-12,
    # 项目实际 chunking 走 §3.1.2 目录粒度,SIZE/OVERLAP 从未被引用)
    expected_sections = [
        "agent_limits",
        "postgres",
        "milvus",
        "redis",
        "embedding",
        "reranker",
        "llm",
        "retrieval",
        "jwt",
        "api",
        "paths",
    ]
    for name in expected_sections:
        assert hasattr(settings, name), f"settings 缺少段:{name}"

    assert isinstance(settings.ENV, str)


def test_postgres_dsn_format() -> None:
    """postgres.dsn 必须是 SQLAlchemy + psycopg v3 driver 接受的格式。"""
    from config.settings import settings

    dsn = settings.postgres.dsn
    assert dsn.startswith("postgresql+psycopg://")
    assert f"@{settings.postgres.HOST}:{settings.postgres.PORT}/" in dsn
    assert dsn.endswith(f"/{settings.postgres.DB}")


# ────────────────────────────────────────────────────────────────────────────
# §8.3 A3 验收:必填项缺失抛错
# ────────────────────────────────────────────────────────────────────────────


def test_llm_api_key_required_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A3 验收:缺失 LLM_API_KEY 必须抛 ValidationError,不允许 fallback 静默继续。"""
    from pydantic import ValidationError

    from config.settings import LLMSettings

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_VISION_API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        LLMSettings(_env_file=None)  # 不读 .env,只看进程 env;模拟生产环境完全缺失

    assert "API_KEY" in str(exc_info.value)


def test_llm_vision_api_key_required_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主链路有 key 但 vision 缺 key 时,LLMSettings 仍必须报错 — F2.5/F9 报告解析必需。"""
    from pydantic import ValidationError

    from config.settings import LLMSettings

    monkeypatch.setenv("LLM_API_KEY", "sk-main-test")
    monkeypatch.delenv("LLM_VISION_API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        LLMSettings(_env_file=None)

    assert "VISION_API_KEY" in str(exc_info.value)


def test_llm_vision_defaults_match_dashscope_qwen() -> None:
    """vision 三件套默认值锁:DashScope + qwen3.5-plus(F2.5/F9 多模态用)。"""
    from config.settings import LLMSettings

    fields = LLMSettings.model_fields
    assert (
        fields["VISION_BASE_URL"].default
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert fields["VISION_MODEL_NAME"].default == "qwen3.5-plus"


def test_llm_main_chain_defaults_match_deepseek() -> None:
    """主链路默认值锁:DeepSeek + deepseek-v4-pro(廉价文本推理)。"""
    from config.settings import LLMSettings

    fields = LLMSettings.model_fields
    assert fields["BASE_URL"].default == "https://api.deepseek.com/v1"
    assert fields["MODEL_NAME"].default == "deepseek-v4-pro"
