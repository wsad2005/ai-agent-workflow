"""平台层测试：规划器 / 执行引擎 / 状态机 / 持久化 / 确认 / 追踪。

AI 相关测试全部使用 httpx.MockTransport，绝不真实调用付费 API。
"""

import json

import httpx
import pytest

from ai_agent_workflow.config import load_settings
from ai_agent_workflow.executor import execute_plan
from ai_agent_workflow.planner import (
    AgentPlan,
    LLMPlanner,
    PlanStep,
    RuleBasedPlanner,
    build_planner,
)
from ai_agent_workflow.storage.task_store import TaskStore, assert_transition
from ai_agent_workflow.tools import Tool, default_registry

# ================================================================ 规划器


def test_rule_planner_math() -> None:
    plan = RuleBasedPlanner().plan("计算 2+3*4 等于多少？", default_registry().tools())
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].arguments == {"expression": "2+3*4"}


def test_rule_planner_time() -> None:
    plan = RuleBasedPlanner().plan("现在几点了？", default_registry().tools())
    assert plan.steps[0].tool_name == "current_time"


def test_rule_planner_plain_question_direct_answer() -> None:
    plan = RuleBasedPlanner().plan("你好", default_registry().tools())
    assert plan.steps == ()
    assert plan.direct_answer is not None


def test_rule_planner_skips_tools_not_registered() -> None:
    plan = RuleBasedPlanner().plan("计算 1+1", tools=[])
    assert plan.steps == ()  # calculator 未注册 → 不产生步骤


def test_build_planner_defaults_to_rule() -> None:
    assert isinstance(build_planner(load_settings({})), RuleBasedPlanner)


def _llm_settings(tmp_path) -> object:
    return load_settings(
        {
            "DATA_DIR": str(tmp_path / "data"),
            "LOG_LEVEL": "WARNING",
            "PLANNER_MODE": "llm",
            "LLM_BASE_URL": "https://ai.example.invalid/v1",  # 测试用假地址
            "LLM_API_KEY": "dummy-ai-key-abcdef",  # 测试用假密钥
            "GENERATION_MODEL": "dummy-model",  # 测试用假模型名
        }
    )


def _llm_planner(content: str, tmp_path) -> LLMPlanner:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert "dummy-ai-key-abcdef" not in str(request.url)  # 密钥只在请求头
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
        )

    return LLMPlanner(_llm_settings(tmp_path), transport=httpx.MockTransport(handler))


def test_llm_planner_valid_plan(tmp_path) -> None:
    content = json.dumps(
        {
            "steps": [
                {"tool_name": "calculator", "arguments": {"expression": "1+1"}},
                {"tool_name": "current_time", "arguments": {}},
            ],
            "direct_answer": None,
        }
    )
    plan = _llm_planner(content, tmp_path).plan("算一下时间", default_registry().tools())
    assert [step.tool_name for step in plan.steps] == ["calculator", "current_time"]


def test_llm_planner_direct_answer(tmp_path) -> None:
    plan = _llm_planner('{"steps": [], "direct_answer": "你好呀"}', tmp_path).plan(
        "你好", default_registry().tools()
    )
    assert plan.steps == ()
    assert plan.direct_answer == "你好呀"


def test_llm_planner_unknown_tool_rejected(tmp_path) -> None:
    bad = '{"steps": [{"tool_name": "hack_tool", "arguments": {}}]}'
    with pytest.raises(Exception) as excinfo:
        _llm_planner(bad, tmp_path).plan("目标", default_registry().tools())
    assert excinfo.value.code == "planning_error"
    assert "hack_tool" in str(excinfo.value)


def test_llm_planner_unknown_argument_rejected(tmp_path) -> None:
    bad = '{"steps": [{"tool_name": "calculator", "arguments": {"bad": "1"}}]}'
    with pytest.raises(Exception) as excinfo:
        _llm_planner(bad, tmp_path).plan("目标", default_registry().tools())
    assert "未知参数" in str(excinfo.value)


def test_llm_planner_missing_argument_rejected(tmp_path) -> None:
    bad = '{"steps": [{"tool_name": "calculator", "arguments": {}}]}'
    with pytest.raises(Exception) as excinfo:
        _llm_planner(bad, tmp_path).plan("目标", default_registry().tools())
    assert "缺少" in str(excinfo.value)


def test_llm_planner_invalid_json_rejected(tmp_path) -> None:
    with pytest.raises(Exception) as excinfo:
        _llm_planner("完全不是 JSON", tmp_path).plan("目标", default_registry().tools())
    assert excinfo.value.code == "planning_error"


# ================================================================ 执行引擎


def _run(plan: AgentPlan, **kwargs) -> list:
    kwargs.setdefault("registry", default_registry())
    return list(execute_plan(plan, **kwargs))


def test_executor_success_event() -> None:
    plan = AgentPlan(steps=(PlanStep("calculator", {"expression": "1+1"}),))
    events = _run(plan)
    assert len(events) == 1
    assert events[0].kind == "step_result"
    assert events[0].status == "success"
    assert events[0].output == "2"


def test_executor_confirmation_required_for_side_effect() -> None:
    plan = AgentPlan(steps=(PlanStep("write_note", {"name": "demo", "content": "hi"}),))
    events = _run(plan)
    assert len(events) == 1
    assert events[0].kind == "confirmation_required"
    assert events[0].status == "waiting"


def test_executor_runs_approved_side_effect(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # write_note 写入相对 data/notes
    plan = AgentPlan(steps=(PlanStep("write_note", {"name": "demo", "content": "hi"}),))
    events = _run(plan, approved_steps=frozenset({0}))
    assert events[0].status == "success"
    assert "已写入" in (events[0].output or "")


def test_executor_unknown_tool_fails_plan() -> None:
    plan = AgentPlan(steps=(PlanStep("ghost", {}),))
    events = _run(plan)
    assert events[0].status == "failed"
    assert "未注册" in (events[0].error or "")


def test_executor_timeout_event(tmp_path, monkeypatch) -> None:
    import time

    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    registry.register(
        Tool(
            name="slow",
            description="慢工具（用于超时测试）",
            parameters={},
            func=lambda arguments: time.sleep(0.5) or "done",
            timeout_seconds=0.2,
        )
    )
    events = _run(AgentPlan(steps=(PlanStep("slow", {}),)), registry=registry, max_retries=0)
    assert [event.status for event in events] == ["timeout"]
    assert "超时" in (events[0].error or "")


def test_executor_timeout_retry_attempts(tmp_path, monkeypatch) -> None:
    import time

    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    registry.register(
        Tool(
            name="slow",
            description="慢工具（用于重试测试）",
            parameters={},
            func=lambda arguments: time.sleep(0.5) or "done",
            timeout_seconds=0.2,
        )
    )
    events = _run(
        AgentPlan(steps=(PlanStep("slow", {}),)), registry=registry, max_retries=2
    )
    assert [event.attempt for event in events] == [1, 2, 3]
    assert all(event.status == "timeout" for event in events)


def test_executor_max_steps_truncates() -> None:
    plan = AgentPlan(
        steps=(
            PlanStep("calculator", {"expression": "1+1"}),
            PlanStep("calculator", {"expression": "2+2"}),
            PlanStep("calculator", {"expression": "3+3"}),
        )
    )
    events = _run(plan, max_steps=2)
    assert len(events) == 2


def test_executor_cancel_between_steps() -> None:
    plan = AgentPlan(
        steps=(
            PlanStep("calculator", {"expression": "1+1"}),
            PlanStep("calculator", {"expression": "2+2"}),
        )
    )
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 1  # 第二个步骤前取消

    events = _run(plan, should_cancel=should_cancel)
    assert events[-1].kind == "cancelled"
    assert events[-1].status == "cancelled"


# ================================================================ 状态机


def test_legal_transitions_pass() -> None:
    assert_transition("pending", "planning")
    assert_transition("planning", "running")
    assert_transition("running", "waiting_confirmation")
    assert_transition("waiting_confirmation", "running")
    assert_transition("running", "succeeded")
    assert_transition("running", "failed")
    assert_transition("pending", "cancelled")


def test_illegal_transitions_raise() -> None:
    import pytest

    from ai_agent_workflow.errors import TaskStateError

    for current, new in (("pending", "running"), ("succeeded", "failed"), ("pending", "succeeded")):
        with pytest.raises(TaskStateError):
            assert_transition(current, new)


def test_terminal_states_cannot_change() -> None:
    import pytest

    from ai_agent_workflow.errors import TaskStateError

    for terminal in ("succeeded", "failed", "cancelled"):
        with pytest.raises(TaskStateError):
            assert_transition(terminal, "running")


# ================================================================ 持久化


def _store(settings) -> TaskStore:
    from ai_agent_workflow.storage.database import Database
    from ai_agent_workflow.storage.migrations import run_migrations

    run_migrations(settings)
    return TaskStore(Database(settings))


def test_task_persisted_and_recovered(settings) -> None:
    store = _store(settings)
    task = store.create("整理求职作品集")
    assert task.status == "pending"

    recovered = store.get(task.id)
    assert recovered is not None
    assert recovered.input == "整理求职作品集"


def test_plan_persisted_and_recovered(settings) -> None:
    store = _store(settings)
    task = store.create("计算 1+1")
    plan = AgentPlan(steps=(PlanStep("calculator", {"expression": "1+1"}),))
    store.transition(task.id, "planning")
    store.save_plan(task.id, plan.to_dict(), planner="rule")

    # 模拟服务重启：全新 TaskStore 实例读取同一数据库
    fresh_store = TaskStore(store._database)  # noqa: SLF001 - 测试访问同容器
    recovered = fresh_store.get(task.id)
    assert recovered is not None
    restored = AgentPlan.from_dict(recovered.plan or {})
    assert restored.steps[0].arguments == {"expression": "1+1"}


def test_step_result_recorded(settings) -> None:
    store = _store(settings)
    task = store.create("计算 1+1")
    store.record_step_result(task.id, 0, "success", "2", None, 1)
    recovered = store.get(task.id)
    assert recovered is not None
    assert recovered.observations[0]["output"] == "2"
    assert recovered.current_step == 1


def test_record_decision_requires_waiting(settings) -> None:
    import pytest

    from ai_agent_workflow.errors import TaskStateError

    store = _store(settings)
    task = store.create("目标")
    with pytest.raises(TaskStateError):
        store.record_decision(task.id, 0, "approved", "")


# ================================================================ 追踪


def test_trace_written_and_read(settings) -> None:
    from pathlib import Path

    from ai_agent_workflow.trace import TraceRecorder

    recorder = TraceRecorder(Path(settings.data_dir) / "traces.jsonl")
    recorder.append(
        task_id="task-1",
        trace_id="t-1",
        action="step:0",
        tool_name="calculator",
        status="success",
        duration_ms=3,
        arguments={"expression": "1+1"},
    )
    records = recorder.read_by_task("task-1")
    assert len(records) == 1
    assert records[0]["tool_name"] == "calculator"
    assert records[0]["arguments"] == {"expression": "1+1"}


def test_trace_redacts_sensitive_values() -> None:
    from ai_agent_workflow.trace import redact_mapping

    redacted = redact_mapping(
        {"api_key": "super-secret", "token": "abc", "expression": "1+1", "password": "pw"}
    )
    assert redacted["api_key"] == "***"
    assert redacted["token"] == "***"
    assert redacted["password"] == "***"
    assert redacted["expression"] == "1+1"


def test_trace_missing_file_returns_empty(settings) -> None:
    from pathlib import Path

    from ai_agent_workflow.trace import TraceRecorder

    assert TraceRecorder(Path(settings.data_dir) / "no-such.jsonl").read_all() == []


# ================================================================ API 端到端


def test_api_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_tools_list_includes_permissions(client) -> None:
    response = client.get("/api/tools")
    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert tools["calculator"]["permission"] == "readonly"
    assert tools["write_note"]["permission"] == "side_effect"
    assert tools["write_note"]["timeout_seconds"] == 10.0


def test_api_task_lifecycle_success(client) -> None:
    created = client.post("/api/tasks", json={"input": "计算 2+3*4 等于多少？"})
    assert created.status_code == 201
    task_id = created.json()["id"]
    assert created.json()["status"] == "pending"

    executed = client.post(f"/api/tasks/{task_id}/execute")
    assert executed.status_code == 200
    body = executed.json()
    assert body["status"] == "succeeded"
    assert body["planner"] == "rule"
    assert "14" in (body["result"] or "")

    traces = client.get(f"/api/tasks/{task_id}/traces").json()
    assert traces["task_id"] == task_id
    actions = [record["action"] for record in traces["records"]]
    assert "plan" in actions and any(action.startswith("step:") for action in actions)


def test_api_execute_terminal_task_rejected(client) -> None:
    task_id = client.post("/api/tasks", json={"input": "你好"}).json()["id"]
    client.post(f"/api/tasks/{task_id}/execute")
    response = client.post(f"/api/tasks/{task_id}/execute")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_state"


def test_api_cancel_waiting_task(client) -> None:
    task_id = client.post("/api/tasks", json={"input": "你好"}).json()["id"]
    client.post(f"/api/tasks/{task_id}/execute")  # 直答 → succeeded
    # 新建一个待执行任务并取消
    other = client.post("/api/tasks", json={"input": "另一个目标"}).json()
    response = client.post(f"/api/tasks/{other['id']}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_api_cancel_terminal_task_rejected(client) -> None:
    task_id = client.post("/api/tasks", json={"input": "你好"}).json()["id"]
    client.post(f"/api/tasks/{task_id}/execute")  # 直答 → succeeded（终态）
    response = client.post(f"/api/tasks/{task_id}/cancel")
    assert response.status_code == 409


def test_api_task_list(client) -> None:
    client.post("/api/tasks", json={"input": "任务一"})
    client.post("/api/tasks", json={"input": "任务二"})
    response = client.get("/api/tasks")
    assert response.status_code == 200
    assert response.json()["total"] >= 2
