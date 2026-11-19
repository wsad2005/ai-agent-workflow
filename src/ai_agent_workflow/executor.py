"""执行引擎：把 AgentPlan 变成可观测、可控的执行过程（事件流）。

设计：
- execute_plan 是一个生成器，逐事件产出执行过程（成功/失败/超时/需确认/取消），
  由上层（服务/API）负责持久化与状态转换——引擎本身不依赖数据库；
- 每个步骤在独立线程中带超时执行（registry.execute）；
- 有副作用的工具在未获人工确认时产出 confirmation_required 事件并暂停；
- 失败重试仅针对超时（业务性失败重试同样输入没有意义）；
- 超时后放弃等待结果，工具线程仍在后台运行至自然结束（当前内置工具均为
  瞬时操作；该限制已在文档中如实说明）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from .errors import ToolExecutionError, ToolNotFoundError
from .planner import AgentPlan
from .tools import ToolRegistry

StepEventKind = Literal["step_result", "confirmation_required", "cancelled"]


@dataclass(frozen=True)
class StepEvent:
    """执行过程中的一个事件。"""

    kind: StepEventKind
    index: int  # 计划中的步骤序号（0 基）
    tool_name: str
    arguments: dict[str, str]
    status: str | None = None  # success | failed | timeout | waiting | cancelled
    output: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    attempt: int | None = None


def execute_plan(
    plan: AgentPlan,
    registry: ToolRegistry,
    *,
    max_steps: int = 10,
    max_retries: int = 1,
    approved_steps: frozenset[int] = frozenset(),
    should_cancel: Callable[[], bool] = lambda: False,
) -> Iterator[StepEvent]:
    """按计划逐步执行并产出事件流。

    - 超过 max_steps 的步骤被丢弃；
    - 每步执行前检查 should_cancel()，命中则产出 cancelled 事件并结束；
    - 有副作用的工具未获确认时产出 confirmation_required 并暂停；
    - 未知工具或业务失败 → failed 事件并终止计划；超时按 max_retries 重试。
    """
    for index, step in enumerate(plan.steps[:max_steps]):
        if should_cancel():
            yield StepEvent(
                kind="cancelled",
                index=index,
                tool_name=step.tool_name,
                arguments=dict(step.arguments),
                status="cancelled",
            )
            return

        try:
            tool = registry.get(step.tool_name)
        except ToolNotFoundError as exc:
            yield StepEvent(
                kind="step_result",
                index=index,
                tool_name=step.tool_name,
                arguments=dict(step.arguments),
                status="failed",
                error=str(exc),
                attempt=0,
            )
            return

        if tool.permission == "side_effect" and index not in approved_steps:
            yield StepEvent(
                kind="confirmation_required",
                index=index,
                tool_name=step.tool_name,
                arguments=dict(step.arguments),
                status="waiting",
            )
            return

        attempts = max(1, max_retries + 1)
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"agent-step-{index}")
            try:
                future = pool.submit(registry.execute, step.tool_name, dict(step.arguments))
                output = future.result(timeout=tool.timeout_seconds)
            except FuturesTimeout:
                yield StepEvent(
                    kind="step_result",
                    index=index,
                    tool_name=step.tool_name,
                    arguments=dict(step.arguments),
                    status="timeout",
                    error=f"工具超时（>{tool.timeout_seconds:g}s），已放弃等待结果",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    attempt=attempt,
                )
                continue
            except ToolExecutionError as exc:
                yield StepEvent(
                    kind="step_result",
                    index=index,
                    tool_name=step.tool_name,
                    arguments=dict(step.arguments),
                    status="failed",
                    error=str(exc),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    attempt=attempt,
                )
                break  # 业务性失败不重试
            except ToolNotFoundError as exc:  # pragma: no cover - get 已提前校验
                yield StepEvent(
                    kind="step_result",
                    index=index,
                    tool_name=step.tool_name,
                    arguments=dict(step.arguments),
                    status="failed",
                    error=str(exc),
                    attempt=attempt,
                )
                break
            else:
                yield StepEvent(
                    kind="step_result",
                    index=index,
                    tool_name=step.tool_name,
                    arguments=dict(step.arguments),
                    status="success",
                    output=output,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    attempt=attempt,
                )
                break
            finally:
                pool.shutdown(wait=False)
        else:
            # 超时重试耗尽：整个计划失败
            return


# concurrent.futures.TimeoutError 在 3.11+ 是内置 TimeoutError 的别名
FuturesTimeout = TimeoutError
