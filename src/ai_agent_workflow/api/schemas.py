"""API 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    app_version: str
    environment: str
    database: str


class TaskCreateRequest(BaseModel):
    """创建 Agent 任务请求。"""

    input: str = Field(min_length=1, max_length=2000, description="任务目标描述")


class ApprovalRequest(BaseModel):
    """人工审批请求。"""

    decision: Literal["approved", "rejected"]
    reason: str = Field(default="", max_length=500)


class AgentTaskOut(BaseModel):
    """Agent 任务响应。"""

    model_config = {"from_attributes": True}

    id: str
    input: str
    status: str
    planner: str
    plan: dict[str, Any] | None
    result: str | None
    failure_reason: str | None
    current_step: int
    observations: list[dict[str, Any]]
    approved_steps: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class TaskListResponse(BaseModel):
    """任务列表响应。"""

    total: int
    items: list[AgentTaskOut]


class ToolOut(BaseModel):
    """工具信息响应。"""

    name: str
    description: str
    parameters: dict[str, str]
    permission: str
    timeout_seconds: float


class TraceResponse(BaseModel):
    """追踪记录响应。"""

    task_id: str
    records: list[dict[str, Any]]
