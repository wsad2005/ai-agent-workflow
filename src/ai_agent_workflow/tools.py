"""工具层：Tool 定义、注册表与内置安全工具。

工具是 Agent 的「手」：模型只负责决定调用哪个工具、传什么参数；
执行与安全边界都在本模块内（计算器使用 AST 白名单求值，绝不使用 eval）。
"""

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .errors import AgentError, ToolExecutionError, ToolNotFoundError

_MAX_EXPRESSION_LENGTH = 200
_MAX_EXPONENT = 1000

_ALLOWED_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass(frozen=True)
class Tool:
    """一个可被 Agent 调用的工具。"""

    name: str
    description: str
    parameters: dict[str, str]  # 参数名 → 说明（下一阶段映射为 JSON Schema）
    func: Callable[[dict[str, str]], str]


class ToolRegistry:
    """工具注册表：注册、列举与执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具；重名视为程序错误。"""
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册：{tool.name}")
        self._tools[tool.name] = tool

    def tools(self) -> list[Tool]:
        """返回全部已注册工具。"""
        return list(self._tools.values())

    def execute(self, name: str, arguments: dict[str, str]) -> str:
        """执行指定工具；未注册抛 ToolNotFoundError，失败抛 ToolExecutionError。"""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"未注册的工具：{name}")
        try:
            return tool.func(arguments)
        except AgentError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"工具 {name} 执行失败：{exc}") from exc


def _evaluate(node: ast.expr) -> float:
    """递归求值 AST 节点；只允许数字常量与白名单运算。"""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("指数过大，已拒绝计算")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"不支持的表达式元素：{type(node).__name__}")


def calculate(arguments: dict[str, str]) -> str:
    """内置工具：安全计算四则运算表达式（AST 白名单，不使用 eval）。"""
    expression = arguments.get("expression", "").strip()
    if not expression:
        raise ToolExecutionError("缺少参数 expression")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolExecutionError("表达式过长")
    try:
        result = _evaluate(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ToolExecutionError(f"表达式无法计算：{exc}") from exc
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def current_time(arguments: dict[str, str]) -> str:
    """内置工具：返回本机当前时间（ISO 8601，含时区）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def default_registry() -> ToolRegistry:
    """返回预装内置工具的注册表。"""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="计算四则运算表达式",
            parameters={"expression": "算式字符串，例如 2+3*4"},
            func=calculate,
        )
    )
    registry.register(
        Tool(
            name="current_time",
            description="返回本机当前时间（ISO 8601）",
            parameters={},
            func=current_time,
        )
    )
    return registry
