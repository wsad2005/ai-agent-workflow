"""Agent 任务接口：创建 / 列表 / 详情 / 执行 / 取消 / 审批 / 追踪。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from ai_agent_workflow.api.deps import PlatformService
from ai_agent_workflow.api.schemas import (
    AgentTaskOut,
    ApprovalRequest,
    TaskCreateRequest,
    TaskListResponse,
    TraceResponse,
)

router = APIRouter()


@router.post(
    "",
    response_model=AgentTaskOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建 Agent 任务（pending，需调用 execute 才会执行）",
)
def create_task(data: TaskCreateRequest, service: PlatformService) -> AgentTaskOut:
    """创建任务；随后调用 execute 触发规划与执行。"""
    return AgentTaskOut.model_validate(service.create_task(data.input))


@router.get(
    "",
    response_model=TaskListResponse,
    summary="任务列表（按创建时间倒序）",
)
def list_tasks(
    service: PlatformService,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> TaskListResponse:
    """返回最近的任务。"""
    tasks = service.list_tasks(limit)
    return TaskListResponse(
        total=len(tasks), items=[AgentTaskOut.model_validate(task) for task in tasks]
    )


@router.get(
    "/{task_id}",
    response_model=AgentTaskOut,
    summary="任务详情",
    responses={404: {"description": "任务不存在"}},
)
def get_task(task_id: str, service: PlatformService) -> AgentTaskOut:
    """返回任务当前状态、计划、结果与执行过程。"""
    return AgentTaskOut.model_validate(service.get_task(task_id))


@router.post(
    "/{task_id}/execute",
    response_model=AgentTaskOut,
    summary="执行任务（规划 → 工具步骤 → 最终回答；需人工确认时暂停）",
    responses={
        404: {"description": "任务不存在"},
        409: {"description": "非法状态转换（如已终结的任务再次执行）"},
        502: {"description": "规划失败"},
    },
)
def execute_task(task_id: str, service: PlatformService) -> AgentTaskOut:
    """同步执行到终态或等待确认；返回执行后的任务状态。"""
    return AgentTaskOut.model_validate(service.execute_task(task_id))


@router.post(
    "/{task_id}/cancel",
    response_model=AgentTaskOut,
    summary="取消任务",
    responses={
        404: {"description": "任务不存在"},
        409: {"description": "任务已终结，不能取消"},
    },
)
def cancel_task(task_id: str, service: PlatformService) -> AgentTaskOut:
    """取消非终态任务；运行中的任务会在下一个步骤边界生效。"""
    return AgentTaskOut.model_validate(service.cancel_task(task_id))


@router.post(
    "/{task_id}/approvals",
    response_model=AgentTaskOut,
    summary="人工审批（批准恢复执行 / 拒绝并记录原因）",
    responses={
        404: {"description": "任务不存在"},
        409: {"description": "任务不处于等待确认状态"},
    },
)
def approve_task(
    task_id: str,
    data: ApprovalRequest,
    service: PlatformService,
) -> AgentTaskOut:
    """批准后从等待的步骤恢复执行；拒绝则任务失败并记录原因。"""
    return AgentTaskOut.model_validate(service.decide_step(task_id, data.decision, data.reason))


@router.get(
    "/{task_id}/traces",
    response_model=TraceResponse,
    summary="任务执行追踪（JSONL 记录，敏感字段已脱敏）",
)
def task_traces(task_id: str, service: PlatformService) -> TraceResponse:
    """返回该任务的全部追踪记录。"""
    service.get_task(task_id)  # 确认任务存在
    return TraceResponse(task_id=task_id, records=service.get_traces(task_id))
