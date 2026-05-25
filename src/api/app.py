"""FastAPI 应用骨架(DEV_SPEC §8.4 G1)。

职责:
- `create_app()` 工厂函数 → 创建 FastAPI 实例
- 注册 `prometheus-fastapi-instrumentator` 暴露 `/metrics`(§5.2.1 ② HTTP 层指标)
- 调 `register_routers(app)` 挂载 G2-G6 业务路由(目前为空,留挂载点)
- 模块级 `app = create_app()` 让 uvicorn 可以 `uvicorn src.api.app:app` 启动

**本任务 NOT-IN-SCOPE**(对应 spec):
- 业务指标埋点(`src/common/metrics.py` 6 个指标)→ H2 完成,G1 只走 instrumentator
  自动采集 HTTP 层
- `/healthz` + `/readyz` 健康检查端点 → H8 完成,**G1 不实现也不占用 `/health` 命名**
- 业务路由(auth / diagnose / patient / admin)→ G2-G6 各自任务
- JWT 中间件 / 限流中间件 → G2 / G3
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from config.logging_config import configure_logging
from config.settings import settings
from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.api.middleware.trace_id import TraceIdMiddleware
from src.api.routes import register_routers
from src.db.postgres.connection import get_engine
from src.db.postgres.metrics import install_engine_metrics


_API_TITLE = "Agentic RAG Diagnose Assistant API"
_API_VERSION = "0.1.0"

# /ui 静态前端目录(portfolio 演示用,见 ui/index.html)。同源挂载避免 CORS,
# 也让前端可以直接 `fetch('/auth/login')` 而不用配 base URL。
_UI_DIR = Path(__file__).resolve().parents[2] / "ui"

# §5.2.1 ② 末:`/metrics` 端点排除 `/healthz`/`/readyz`/`/metrics` 自身避免自污染。
# G1 还没有 /healthz /readyz(H8 才建),但 instrumentator 配置里先列上,等 H8
# 实现时无需回头改本文件 —— 防御式排除,字符串列表比"将来记得改"靠谱。
_METRICS_EXCLUDED_HANDLERS = ["/healthz", "/readyz", "/metrics"]


def create_app() -> FastAPI:
    """构造并配置 FastAPI 实例。测试用 `from src.api.app import create_app; app = create_app()`
    可绕过模块级单例,得到独立 app(便于隔离 fixture)。"""
    # 全局 JSON 日志(H4 / spec §5.2.1.1)— 必须在创建 FastAPI 之前调,否则
    # uvicorn 已挂上的纯文本 handler 没机会被替换。幂等。
    configure_logging()

    # PG 依赖层指标(H2 / spec §5.2.1 ③)— SQLAlchemy event listener,幂等
    install_engine_metrics(get_engine())

    app = FastAPI(title=_API_TITLE, version=_API_VERSION)

    # HTTP 层指标自动采集(§5.2.1 ②)。一行接入,业务代码完全不感知。
    # H2 任务来时会扩展业务指标,但 instrumentator 这部分不会再动。
    Instrumentator(
        excluded_handlers=_METRICS_EXCLUDED_HANDLERS,
        should_group_status_codes=False,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # 限流(G3)— 内存滑动窗口,H6 改 Redis 后端时换 RateLimitBackend 实现即可。
    # 必须在 register_routers 之前挂,确保对所有业务路由生效;Instrumentator 的
    # /metrics 端点已在内部加入 excluded_paths 不会被挡。
    app.add_middleware(
        RateLimitMiddleware,
        limit=settings.api.RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )

    # trace_id 注入(H4 / spec §5.2.1.1)— 必须在 RateLimitMiddleware **之后**
    # 调 add_middleware(Starlette 中间件挂载是反序执行,后挂的更外层),让
    # 429 响应也带 trace_id。所有日志、rag_trace、Loki label 用同一 ID。
    app.add_middleware(TraceIdMiddleware)

    register_routers(app)

    # /ui 前端(演示用,可选;目录不存在则跳过以免开发期没建报错)
    if _UI_DIR.is_dir():
        app.mount(
            "/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui"
        )

    return app


# uvicorn 入口:`uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
app = create_app()
