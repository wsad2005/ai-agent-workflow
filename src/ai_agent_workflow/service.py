"""Agent 平台服务：创建任务、规划、执行、人工确认、取消与追踪。

编排关系：TaskStore（持久化+状态机）← 本服务 ← API/CLI；
ExecutionEngine 只产出事件流，持久化与状态转换全部在本层完成。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Literal

import httpx

from ai_agent_workflow.config import Settings
from ai_agent_workflow.errors import EmptyInputError, NotFoundError, TaskStateError
from ai_agent_workflow.executor import StepEvent, execute_plan
from ai_agent_workflow.logging_setup import get_logger
from ai_agent_workflow.planner import AgentPlan, build_planner
from ai_agent_workflow.storage.database import Database
from ai_agent_workflow.storage.models import AgentTask
from ai_agent_workflow.storage.task_store import TERMINAL_STATUSES, TaskStore
from ai_agent_workflow.tools import ToolRegistry, default_registry
from ai_agent_workflow.trace import TraceRecorder

logger = get_logger(__name__)


class AgentPlatformService:
    """平台编排服务：把规划器、执行引擎、持久化与追踪串成完整任务生命周期。"""

    def __init__(
        self,
        settings: Settings,
        database: Database | None = None,
        *,
        registry: ToolRegistry | None = None,
        planner: object | None = None,
        trace_recorder: TraceRecorder | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._database = database or Database(settings)
        self._registry = registry or default_registry()
        self._store = TaskStore(self._database)
        self._planner = planner or build_planner(settings, transport=transport)
        self._trace = trace_recorder or TraceRecorder(settings.data_dir / "traces.jsonl")
        self._cancel_requested: set[str] = set()

    # ------------------------------------------------------------------ 查询
    def create_task(self, task_input: str) -> AgentTask:
        """创建任务（pending），等待显式执行。"""
        text = (task_input or "").strip()
        if not text:
            raise EmptyInputError("任务输入为空")
        task = self._store.create(text)
        self._trace.append(
            task_id=task.id, trace_id=self._trace_id(), action="create", status="pending"
        )
        return task

    def get_task(self, task_id: str) -> AgentTask:
        """获取任务；不存在抛 NotFoundError。"""
        task = self._store.get(task_id)
        if task is None:
            raise NotFoundError(f"任务不存在：{task_id}")
        return task

    def list_tasks(self, limit: int = 50) -> list[AgentTask]:
        """返回最近任务列表。"""
        return self._store.list(limit)

    def get_traces(self, task_id: str) -> list[dict]:
        """返回指定任务的追踪记录（已脱敏）。"""
        return self._trace.read_by_task(task_id)

    # ------------------------------------------------------------------ 执行
    def execute_task(self, task_id: str) -> AgentTask:
        """规划（如需）并执行任务；可在等待确认后恢复继续执行。"""
        task = self._store.get(task_id)
        if task is None:
            raise NotFoundError(f"任务不存在：{task_id}")
        if task.status in TERMINAL_STATUSES:
            raise TaskStateError(f"任务已终结（{task.status}），不能再次执行")
        trace_id = self._trace_id()

        if task.status == "pending":
            self._store.transition(task_id, "planning")
            plan = self._plan(task_id, trace_id, task.input)
        elif task.status == "waiting_confirmation":
            plan = AgentPlan.from_dict(task.plan or {})
        else:
            raise TaskStateError(f"任务当前状态 {task.status} 不能执行")

        self._store.transition(task_id, "running")
        approved = self._store.approved_indexes(self._store.get(task_id))
        outputs: list[tuple[str, str]] = []

        events: Iterator[StepEvent] = execute_plan(
            plan,
            self._registry,
            approved_steps=approved,
            should_cancel=lambda: task_id in self._cancel_requested,
        )
        for event in events:
            self._trace_event(task_id, trace_id, event)
            if event.kind == "cancelled":
                self._store.transition(task_id, "cancelled")
                self._store.set_result(task_id, "任务已取消")
                return self._store.get(task_id)
            if event.kind == "confirmation_required":
                self._store.transition(task_id, "waiting_confirmation")
                self._store.set_current_step(task_id, event.index)
                return self._store.get(task_id)
            self._store.record_step_result(
                task_id, event.index, event.status or "", event.output, event.error, event.attempt
            )
            if event.status in ("failed", "timeout"):
                self._store.transition(task_id, "failed")
                self._store.set_result(task_id, None, failure_reason=event.error)
                return self._store.get(task_id)
            if event.status == "success":
                outputs.append((event.tool_name, event.output or ""))

        final = plan.direct_answer or self._compose_answer(outputs)
        self._store.transition(task_id, "succeeded")
        self._store.set_result(task_id, final)
        self._trace.append(
            task_id=task_id, trace_id=trace_id, action="execute_end", status="succeeded"
        )
        return self._store.get(task_id)

    # ------------------------------------------------------------------ 取消
    def cancel_task(self, task_id: str, reason: str = "") -> AgentTask:
        """取消任务：非运行状态立即取消；运行中标记，由执行循环在步骤间生效。"""
        task = self._store.get(task_id)
        if task is None:
            raise NotFoundError(f"任务不存在：{task_id}")
        if task.status in TERMINAL_STATUSES:
            raise TaskStateError(f"任务已终结（{task.status}），不能取消")
        if task.status == "running":
            self._cancel_requested.add(task_id)
            logger.info("已请求取消运行中的任务 %s", task_id)
            return task
        self._store.transition(task_id, "cancelled")
        message = f"任务已取消：{reason}" if reason else "任务已取消"
        self._store.set_result(task_id, message)
        self._trace.append(
            task_id=task_id, trace_id=self._trace_id(), action="cancel", status="cancelled"
        )
        return self._store.get(task_id)

    # ------------------------------------------------------------------ 确认
    def decide_step(
        self, task_id: str, decision: Literal["approved", "rejected"], reason: str = ""
    ) -> AgentTask:
        """人工审批：批准则从当前步骤恢复执行，拒绝则任务失败并记录原因。"""
        task = self._store.get(task_id)
        if task is None:
            raise NotFoundError(f"任务不存在：{task_id}")
        step_index = task.current_step
        self._store.record_decision(task_id, step_index, decision, reason)
        self._trace.append(
            task_id=task_id,
            trace_id=self._trace_id(),
            action="approval",
            tool_name="",
            status=decision,
            error=reason or None,
        )
        if decision == "rejected":
            self._store.transition(task_id, "failed")
            reason_text = reason or '未说明'
            self._store.set_result(
                task_id, None, failure_reason=f'人工拒绝（步骤 {step_index + 1}）：{reason_text}'
            )
            return self._store.get(task_id)
        return self.execute_task(task_id)

    # ------------------------------------------------------------------ 内部
    def _plan(self, task_id: str, trace_id: str, task_input: str) -> AgentPlan:
        """生成计划；失败把任务置为 failed 并原样抛出异常。"""
        try:
            plan = self._planner.plan(task_input, self._registry.tools())
        except Exception as exc:
            self._store.transition(task_id, "failed")
            self._store.set_result(task_id, None, failure_reason=str(exc))
            self._trace.append(
                task_id=task_id, trace_id=trace_id, action="plan", status="failed", error=str(exc)
            )
            raise
        planner_name = getattr(self._planner, 'provider', 'rule')
        self._store.save_plan(task_id, plan.to_dict(), planner=planner_name)
        self._trace.append(
            task_id=task_id,
            trace_id=trace_id,
            action="plan",
            status="ok",
            steps=len(plan.steps),
        )
        return plan

    def _trace_event(self, task_id: str, trace_id: str, event: StepEvent) -> None:
        """把执行事件写入追踪（参数自动脱敏）。"""
        self._trace.append(
            task_id=task_id,
            trace_id=trace_id,
            action=f"step:{event.index}" if event.kind == "step_result" else event.kind,
            tool_name=event.tool_name,
            status=event.status or event.kind,
            duration_ms=event.duration_ms,
            error=event.error,
            arguments=event.arguments,
        )

    def _trace_id(self) -> str:
        """生成一次执行的追踪 id。"""
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _compose_answer(outputs: list[tuple[str, str]]) -> str:
        """把成功步骤的输出整理为最终回答。"""
        if not outputs:
            return "[agent] 计划为空，没有可执行的步骤。"
        lines = [f"[agent] 已完成 {len(outputs)} 个步骤："]
        lines.extend(f"- {name}: {output}" for name, output in outputs)
        return "\n".join(lines)
