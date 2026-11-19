"""依赖注入：应用容器与平台服务。"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ai_agent_workflow.config import Settings
from ai_agent_workflow.planner import build_planner
from ai_agent_workflow.service import AgentPlatformService
from ai_agent_workflow.storage.database import Database
from ai_agent_workflow.tools import default_registry
from ai_agent_workflow.trace import TraceRecorder


class Container:
    """应用级容器：持有配置、数据库、注册表、规划器、追踪与平台服务。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_directories()
        self.database = Database(settings)
        self.registry = default_registry()
        self.planner = build_planner(settings)
        self.trace = TraceRecorder(settings.data_dir / "traces.jsonl")
        self.service = AgentPlatformService(
            settings,
            self.database,
            registry=self.registry,
            planner=self.planner,
            trace_recorder=self.trace,
        )

    def dispose(self) -> None:
        """应用关闭时释放资源。"""
        self.database.dispose()


def get_container(request: Request) -> Container:
    """从 app.state 取应用容器。"""
    return request.app.state.container


def get_platform_service(
    container: Annotated[Container, Depends(get_container)],
) -> AgentPlatformService:
    """平台编排服务。"""
    return container.service


PlatformService = Annotated[AgentPlatformService, Depends(get_platform_service)]
