"""任务仓储与状态机。

状态机规则（``ALLOWED_TRANSITIONS``）在写入前强制校验，非法跳转抛
TaskStateError——pending/planning/waiting_confirmation/running 可取消，
终态（succeeded/failed/cancelled）不可再变。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..errors import NotFoundError, TaskStateError
from .database import Database
from .models import AgentTask, now_local

#: 全部合法状态
TASK_STATUSES = (
    "pending",
    "planning",
    "waiting_confirmation",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)

#: 合法状态转换表：不在表内的跳转一律拒绝
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"planning", "cancelled"},
    "planning": {"running", "failed", "waiting_confirmation", "cancelled"},
    "running": {"succeeded", "failed", "waiting_confirmation", "cancelled"},
    "waiting_confirmation": {"running", "failed", "cancelled"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}

#: 终态集合
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def assert_transition(current: str, new_status: str) -> None:
    """校验状态转换合法性；非法跳转抛 TaskStateError。"""
    if current not in ALLOWED_TRANSITIONS:
        raise TaskStateError(f"未知状态：{current!r}")
    if new_status not in ALLOWED_TRANSITIONS[current]:
        raise TaskStateError(f"非法状态转换：{current} → {new_status}")


class TaskStore:
    """Agent 任务仓储：创建、查询、状态转换与执行过程记录。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create(self, task_input: str) -> AgentTask:
        """创建任务（初始状态 pending）。"""
        with self._database.session() as session:
            task = AgentTask(id=uuid.uuid4().hex, input=task_input, status="pending")
            session.add(task)
            session.flush()
            return task

    def get(self, task_id: str) -> AgentTask | None:
        """按 id 查询任务。"""
        with self._database.session() as session:
            return session.get(AgentTask, task_id)

    def list(self, limit: int = 50) -> list[AgentTask]:
        """按创建时间倒序返回最近任务。"""
        with self._database.session() as session:
            return list(
                session.scalars(
                    select(AgentTask).order_by(AgentTask.created_at.desc()).limit(limit)
                )
            )

    def transition(self, task_id: str, new_status: str) -> AgentTask:
        """状态转换（先校验合法性）。"""
        with self._database.session() as session:
            task = self._get(session, task_id)
            assert_transition(task.status, new_status)
            task.status = new_status
            task.updated_at = now_local()
            session.flush()
            return task

    def save_plan(self, task_id: str, plan: dict, planner: str) -> AgentTask:
        """保存规划结果并记录规划器类型。"""
        with self._database.session() as session:
            task = self._get(session, task_id)
            task.plan = plan
            task.planner = planner
            task.updated_at = now_local()
            session.flush()
            return task

    def set_current_step(self, task_id: str, index: int) -> AgentTask:
        """记录当前执行到的步骤序号（0 基）。"""
        with self._database.session() as session:
            task = self._get(session, task_id)
            task.current_step = index
            task.updated_at = now_local()
            session.flush()
            return task

    def record_step_result(
        self,
        task_id: str,
        index: int,
        status: str,
        output: str | None,
        error: str | None,
        attempt: int | None = None,
    ) -> AgentTask:
        """把步骤结果追加进 observations。"""
        with self._database.session() as session:
            task = self._get(session, task_id)
            observations = list(task.observations or [])
            observations.append(
                {
                    "index": index,
                    "status": status,
                    "output": output,
                    "error": error,
                    "attempt": attempt,
                    "at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                }
            )
            task.observations = observations
            task.current_step = index + 1
            task.updated_at = now_local()
            session.flush()
            return task

    def set_result(
        self, task_id: str, result: str | None, failure_reason: str | None = None
    ) -> AgentTask:
        """写入最终回答或失败原因。"""
        with self._database.session() as session:
            task = self._get(session, task_id)
            task.result = result
            task.failure_reason = failure_reason
            task.updated_at = now_local()
            session.flush()
            return task

    def record_decision(
        self, task_id: str, step_index: int, decision: str, reason: str
    ) -> AgentTask:
        """记录人工确认决定（approved/rejected + 原因）；仅限等待确认状态。"""
        with self._database.session() as session:
            task = self._get(session, task_id)
            if task.status != "waiting_confirmation":
                raise TaskStateError(
                    f"任务当前状态为 {task.status}，只有 waiting_confirmation 可以审批"
                )
            approved = dict(task.approved_steps or {})
            approved[str(step_index)] = {
                "decision": decision,
                "reason": reason,
                "decided_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            }
            task.approved_steps = approved
            task.updated_at = now_local()
            session.flush()
            return task

    def approved_indexes(self, task: AgentTask) -> frozenset[int]:
        """从任务记录中取出已批准的步骤序号集合。"""
        approved = task.approved_steps or {}
        return frozenset(
            int(index)
            for index, decision in approved.items()
            if decision.get("decision") == "approved"
        )

    def _get(self, session: Session, task_id: str) -> AgentTask:
        task = session.get(AgentTask, task_id)
        if task is None:
            raise NotFoundError(f"任务不存在：{task_id}")
        return task
