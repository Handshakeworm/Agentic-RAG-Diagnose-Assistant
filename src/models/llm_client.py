"""src/models/llm_client.py — LLM 推理客户端工厂(DEV_SPEC §8.3 B8)。

只提供配置好的 ChatOpenAI 实例,**不封装** retry / metrics / structured output。
那些都按 DEV_SPEC §9.1 在每个调用站点自己写裸代码 try/except/finally(详见 CLAUDE.md
"LLM call contract" 段)。

业务节点用法(典型):

    from src.models.llm_client import get_llm
    from src.agent.schemas.query_construction import QueryConstructionOutput
    from src.common.metrics import _attempts, _failures, _latency, retry_observer

    llm = get_llm()
    chain = llm.with_structured_output(QueryConstructionOutput).with_retry(stop_after_attempt=3)
    # ... 按 §9.1 模板写 try/except/finally + 6 个指标埋点
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.settings import settings


@lru_cache(maxsize=8)
def get_llm(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.1,
    timeout_seconds: int = 60,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """返回 ChatOpenAI 实例,同参数自动复用(lru_cache)。

    默认走主链路(settings.llm.BASE_URL / API_KEY / MODEL_NAME — DeepSeek)。
    `max_tokens` 默认从 `settings.llm.MAX_TOKENS` 读(.env `LLM_MAX_TOKENS` 可覆盖);
    DeepSeek thinking 模型不设 max_tokens 会拖到 100s+,务必保留这个限制。

    **Vision LLM(⑩ diagnose / report_parser)必须显式覆盖默认 timeout/max_tokens**:
    qwen3.5-plus thinking 模式 reasoning + content 共享 budget,2048 默认会触发
    LengthFinish 截断;60s timeout 会触发 ReadTimeout。评测验证的稳定值:
        get_llm(
            model=settings.llm.VISION_MODEL_NAME,
            base_url=settings.llm.VISION_BASE_URL,
            api_key=settings.llm.VISION_API_KEY,
            timeout_seconds=300,
            max_tokens=16384,
        )
    enrichment 等场景若要换模型/温度/max_tokens,显式传参覆盖。
    """
    return ChatOpenAI(
        model=model or settings.llm.MODEL_NAME,
        base_url=base_url or settings.llm.BASE_URL,
        api_key=api_key or settings.llm.API_KEY,
        temperature=temperature,
        timeout=timeout_seconds,
        max_tokens=max_tokens if max_tokens is not None else settings.llm.MAX_TOKENS,
    )
