"""tests/unit/test_api_app_skeleton.py — G1 启动冒烟测试。

锁住 spec §8.4 G1 验收标准:
- 应用可启动并挂载路由集合(register_routers 被调,目前空)
- `/metrics` 端点存在且返回 Prometheus exposition format
- HTTP 层指标 family 出现在 /metrics(`http_requests_total` 等)
- `/healthz` / `/readyz` **未实现**(spec G1 显式不实现,留给 H8)
- 不存在的路径返回 404 而不是 500

不需要真 PG / Redis / LLM;FastAPI app 本身没业务依赖(G2-G6 的路由还是空的)。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, create_app


@pytest.fixture
def client() -> TestClient:
    """共享模块级 app 即可 — G1 阶段没业务状态需要隔离。"""
    return TestClient(app)


# ────────────────────────────────────────────────────────────────────────────
# 启动 + 工厂
# ────────────────────────────────────────────────────────────────────────────


def test_create_app_returns_independent_instance() -> None:
    """create_app() 工厂可重复调出独立 app(便于隔离 fixture)。"""
    app_a = create_app()
    app_b = create_app()
    assert app_a is not app_b
    assert app_a.title == app_b.title == "Agentic RAG Diagnose Assistant API"


def test_module_level_app_is_fastapi_instance() -> None:
    """uvicorn 入口 `src.api.app:app` 必须存在且是 FastAPI 实例。"""
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)


# ────────────────────────────────────────────────────────────────────────────
# /metrics 端点(spec §5.2.1 ② HTTP 层指标)
# ────────────────────────────────────────────────────────────────────────────


def test_metrics_endpoint_returns_200(client: TestClient) -> None:
    """instrumentator 接入成功 → /metrics 200。"""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus exposition 是 text/plain; version=...
    assert "text/plain" in resp.headers.get("content-type", "")


def test_metrics_endpoint_emits_http_metric_families(client: TestClient) -> None:
    """spec §5.2.1 ②:HTTP 层指标自动出现(`http_requests_total` 等)。
    先发一个普通请求触发计数器,再读 /metrics 断言 family 名出现。"""
    client.get("/some-nonexistent-path")  # 触发一次 404 让计数器 +1

    body = client.get("/metrics").text
    # instrumentator 默认输出的核心 family 名
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body


def test_metrics_endpoint_self_excluded_from_metrics(client: TestClient) -> None:
    """spec §5.2.1 ② 末:`/metrics` 自身排除避免自污染。
    多次打 /metrics 后,计数器里的 handler 不应出现 '/metrics'。"""
    for _ in range(3):
        client.get("/metrics")

    body = client.get("/metrics").text
    # 解析 http_requests_total 行,确认没有 handler="/metrics" 的样本
    metrics_handler_lines = [
        line
        for line in body.splitlines()
        if line.startswith("http_requests_total{") and 'handler="/metrics"' in line
    ]
    assert metrics_handler_lines == [], (
        "/metrics 端点本身被采集进了指标,违反 §5.2.1 ② 自污染规则"
    )


# ────────────────────────────────────────────────────────────────────────────
# G1 NOT-IN-SCOPE 锁定:G2-G6 才实现的业务路由
# (`/healthz` / `/readyz` 已在 H8 实现,test_health_routes.py 覆盖,本文件不再断言)
# ────────────────────────────────────────────────────────────────────────────


def test_root_path_not_implemented(client: TestClient) -> None:
    """没规划顶层欢迎页,长期 404。
    G4-G6 完工后 /diagnose、/patients/me、/admin/* 已实现,不再断言 404。"""
    assert client.get("/").status_code == 404


def test_register_routers_mounts_currently_implemented_routers() -> None:
    """spec G1 留挂载点给 G2-G6;每个 G 阶段任务完成会往 register_routers 解开一行。
    本断言只锁"已挂的不能被退化删掉"。

    当前实现状态:
    - G2 ✅ /auth/register, /auth/login, /auth/me
    - G4/G5/G6 业务路由已实现
    - H8 ✅ /healthz, /readyz
    """
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert {"/auth/register", "/auth/login", "/auth/me"}.issubset(paths), (
        f"G2 auth 路由未全部挂载,当前 paths={paths}"
    )
