"""工具层：Tool 定义、注册表与内置安全工具。

工具是 Agent 的「手」：模型只负责决定调用哪个工具、传什么参数；
执行与安全边界都在本模块内（计算器使用 AST 白名单求值，绝不使用 eval）。

v0.3 升级：工具携带权限级别（只读 / 有副作用）与超时限制；
注册表执行前做参数 Schema 校验，有副作用的工具必须经人工确认。
"""

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from .errors import AgentError, ToolExecutionError, ToolNotFoundError

_MAX_EXPRESSION_LENGTH = 200
_MAX_EXPONENT = 1000

#: 工具权限级别：readonly 无副作用可自动执行；side_effect 必须人工确认
ToolPermission = Literal["readonly", "side_effect"]

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
    """一个可被 Agent 调用的工具。

    parameters 既是参数说明，也是执行前的 Schema（键集合）约束：
    缺少必填键或传入未知键都会被拒绝。
    """

    name: str
    description: str
    parameters: dict[str, str]  # 参数名 → 说明（下一阶段映射为 JSON Schema）
    func: Callable[[dict[str, str]], str]
    permission: ToolPermission = "readonly"
    timeout_seconds: float = 10.0
    tags: tuple[str, ...] = field(default=())


class ToolRegistry:
    """工具注册表：注册、列举、Schema 校验与执行。"""

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

    def get(self, name: str) -> Tool:
        """按名称获取工具；未注册抛 ToolNotFoundError。"""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"未注册的工具：{name}")
        return tool

    def validate_arguments(self, tool: Tool, arguments: dict[str, str]) -> None:
        """按工具的参数 Schema 校验：键集合必须完全一致。"""
        expected = set(tool.parameters)
        provided = set(arguments)
        missing = sorted(expected - provided)
        unknown = sorted(provided - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"缺少参数：{', '.join(missing)}")
            if unknown:
                details.append(f"未知参数：{', '.join(unknown)}")
            raise ToolExecutionError(f"工具 {tool.name} 参数校验失败（{'；'.join(details)}）")

    def execute(self, name: str, arguments: dict[str, str]) -> str:
        """执行指定工具；未注册抛 ToolNotFoundError，失败抛 ToolExecutionError。"""
        tool = self.get(name)
        self.validate_arguments(tool, dict(arguments))
        try:
            return tool.func(dict(arguments))
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


def write_note(arguments: dict[str, str]) -> str:
    """内置工具（有副作用）：把一段文本写入 data/notes/<name>.txt。

    文件名只允许字母/数字/下划线/连字符（防路径穿越）；
    目录不存在时自动创建。需要人工确认后才会执行。
    """
    name = (arguments.get("name") or "").strip()
    content = arguments.get("content", "")
    if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ToolExecutionError("缺少合法参数 name（仅限字母/数字/下划线/连字符）")
    notes_dir = Path("data") / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    target = notes_dir / f"{name}.txt"
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target}（{len(content)} 字符）"


def default_registry() -> ToolRegistry:
    """返回预装内置工具的注册表（只读 + 副作用工具分离权限）。"""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="计算四则运算表达式",
            parameters={"expression": "算式字符串，例如 2+3*4"},
            func=calculate,
            permission="readonly",
            timeout_seconds=10.0,
        )
    )
    registry.register(
        Tool(
            name="current_time",
            description="返回本机当前时间（ISO 8601）",
            parameters={},
            func=current_time,
            permission="readonly",
            timeout_seconds=5.0,
        )
    )
    registry.register(
        Tool(
            name="write_note",
            description="把文本写入 data/notes/<name>.txt（有副作用，需人工确认）",
            parameters={"name": "笔记名称（字母/数字/下划线/连字符）", "content": "笔记内容"},
            func=write_note,
            permission="side_effect",
            timeout_seconds=10.0,
        )
    )
    return registry
