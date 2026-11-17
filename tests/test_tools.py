"""tools 模块测试：注册表行为与内置安全工具。"""

from datetime import datetime

import pytest

from ai_agent_workflow.errors import ToolExecutionError, ToolNotFoundError
from ai_agent_workflow.tools import Tool, ToolRegistry, calculate, current_time, default_registry


def test_calculator_basic_operations() -> None:
    assert calculate({"expression": "2+3*4"}) == "14"
    assert calculate({"expression": "(2+3)*4"}) == "20"
    assert calculate({"expression": "10/4"}) == "2.5"
    assert calculate({"expression": "2**8"}) == "256"
    assert calculate({"expression": "-5+3"}) == "-2"


def test_calculator_rejects_dangerous_input() -> None:
    bad_inputs = (
        "__import__('os').system('echo hi')",
        "os.system('x')",
        "1;2",
        "x=1",
        "'a'+'b'",
        "lambda: 1",
    )
    for bad in bad_inputs:
        with pytest.raises(ToolExecutionError):
            calculate({"expression": bad})


def test_calculator_rejects_huge_exponent() -> None:
    with pytest.raises(ToolExecutionError):
        calculate({"expression": "9**9**9"})


def test_calculator_rejects_too_long_expression() -> None:
    with pytest.raises(ToolExecutionError):
        calculate({"expression": "1+" * 200 + "1"})


def test_calculator_missing_argument() -> None:
    with pytest.raises(ToolExecutionError):
        calculate({})


def test_current_time_returns_parseable_iso() -> None:
    parsed = datetime.fromisoformat(current_time({}))
    assert parsed.tzinfo is not None


def test_registry_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().execute("nope", {})


def test_registry_duplicate_registration_raises() -> None:
    registry = ToolRegistry()
    tool = Tool(name="t", description="d", parameters={}, func=lambda arguments: "ok")
    registry.register(tool)
    with pytest.raises(ValueError):
        registry.register(tool)


def test_default_registry_contains_builtins() -> None:
    names = {tool.name for tool in default_registry().tools()}
    assert {"calculator", "current_time"} <= names


def test_default_registry_executes_calculator() -> None:
    assert default_registry().execute("calculator", {"expression": "1+1"}) == "2"
