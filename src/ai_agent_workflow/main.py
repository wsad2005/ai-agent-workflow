"""FastAPI 应用入口。

启动顺序：读配置 → 确保目录 → 执行数据库迁移（幂等）→ 装配容器与路由。
从不删除或重建数据库。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_agent_workflow.api.handlers import register_exception_handlers
from ai_agent_workflow.api.router import api_router
from ai_agent_workflow.config import Settings, load_settings
from ai_agent_workflow.logging_setup import configure_logging, get_logger
from ai_agent_workflow.storage.migrations import run_migrations

logger = get_logger(__name__)

DESCRIPTION = """
AI Agent 执行平台：把自然语言目标规划为工具调用步骤并安全执行。

**能力**：任务持久化与状态机（pending/planning/waiting_confirmation/running/
succeeded/failed/cancelled）→ 规则式规划（默认）或可选 LLM 规划 → 工具执行
（Schema 校验、超时、重试、权限分级）→ 人工确认（批准/拒绝）→ JSONL 追踪。

**接口文档**：`/docs`（Swagger UI）、`/redoc`、`/openapi.json`。
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """应用工厂；测试通过传入自定义 Settings 使用隔离数据库。"""
    resolved = settings or load_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved.ensure_directories()
        run_migrations(resolved)
        from ai_agent_workflow.api.deps import Container

        container = Container(resolved)
        app.state.container = container
        logger.info("%s v%s 已启动", resolved.app_name, resolved.app_version)
        try:
            yield
        finally:
            container.dispose()

    app = FastAPI(
        title="AI Agent Execution Platform",
        description=DESCRIPTION,
        version=resolved.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    register_exception_handlers(app)
    app.include_router(api_router, prefix=resolved.api_prefix)
    return app


app = create_app()
