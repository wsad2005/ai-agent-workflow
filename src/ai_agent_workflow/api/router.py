"""API 路由汇总。"""

from __future__ import annotations

from fastapi import APIRouter

from ai_agent_workflow.api.routes_health import router as health_router
from ai_agent_workflow.api.routes_tasks import router as tasks_router
from ai_agent_workflow.api.routes_tools import router as tools_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(tasks_router, prefix="/tasks", tags=["Agent 任务"])
api_router.include_router(tools_router, prefix="/tools", tags=["工具"])
