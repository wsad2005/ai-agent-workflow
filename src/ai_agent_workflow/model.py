"""模型层：Agent 的大脑接口、确定性 Mock 与 OpenAI 兼容真实客户端。

AgentModelClient 是模型边界：Mock 用规则模拟「决定是否调用工具」；
real 模式通过 OpenAI 兼容 function calling（已验证目标：智谱 GLM）让
真实 LLM 决策。工具 JSON Schema 从 Tool 定义自动生成，单一事实来源。
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import InvalidResponseError
from .http_transport import post_json
from .tools import Tool

_MATH_CANDIDATE = re.compile(r"[-+*/()0-9.\s]{3,}")
_HAS_DIGIT = re.compile(r"\d")
_HAS_OPERATOR = re.compile(r"[-+*/]")
_TIME_KEYWORDS = ("时间", "几点", "日期")


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用请求；id 由真实 API 返回，Mock 为空串。"""

    name: str
    arguments: dict[str, str]
    id: str = ""


@dataclass(frozen=True)
class Message:
    """对话消息。

    - assistant 消息可用 tool_calls 记录模型本轮请求的全部调用；
    - tool 消息用 tool_name 记录产生它的工具、tool_call_id 对应请求 id。
    """

    role: str  # system | user | assistant | tool
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ModelReply:
    """模型的一轮输出：要么给出最终回答，要么请求调用工具。"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class AgentModelClient(Protocol):
    """模型客户端接口：对话历史 + 可用工具 → 一轮模型输出。"""

    def chat(self, messages: list[Message], tools: Sequence[Tool]) -> ModelReply:
        """根据对话历史与可用工具产生一轮输出。"""
        ...


def _extract_expression(text: str) -> str | None:
    """从文本中提取首个「像算式」的片段；没有则返回 None。"""
    for candidate in _MATH_CANDIDATE.findall(text):
        cleaned = candidate.strip()
        if _HAS_DIGIT.search(cleaned) and _HAS_OPERATOR.search(cleaned):
            return cleaned
    return None


class MockAgentModelClient:
    """确定性 Mock 模型：用规则模拟「模型决定是否调用工具」。

    策略：
    - 最新消息是工具结果 → 引用结果给出最终回答；
    - 用户消息包含可计算的算式 → 请求调用 calculator；
    - 用户消息询问时间/日期 → 请求调用 current_time；
    - 其他 → 直接给出占位回答。
    工具轮数达到上限后不再发起调用，保证收敛。
    """

    def __init__(self, max_tool_rounds: int = 2) -> None:
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds 必须为正整数")
        self._max_tool_rounds = max_tool_rounds

    def chat(self, messages: list[Message], tools: Sequence[Tool]) -> ModelReply:
        """按规则产生一轮输出（real 模式由真实模型替换本实现）。"""
        last = messages[-1]
        if last.role == "tool":
            return ModelReply(content=f"[mock] 工具结果：{last.content}")
        if last.role != "user":
            return ModelReply(content="[mock] 请继续你的问题。")
        rounds_used = sum(1 for message in messages if message.role == "tool")
        if rounds_used >= self._max_tool_rounds:
            return ModelReply(content=f"[mock] 已完成工具调用，回答你的问题：{last.content}")
        expression = _extract_expression(last.content)
        if expression is not None:
            return ModelReply(
                tool_calls=[ToolCall(name="calculator", arguments={"expression": expression})]
            )
        if any(keyword in last.content for keyword in _TIME_KEYWORDS):
            return ModelReply(tool_calls=[ToolCall(name="current_time", arguments={})])
        return ModelReply(content=f"[mock] 已收到你的问题：{last.content}")


def _tool_to_schema(tool: Tool) -> dict[str, object]:
    """把 Tool 定义转换为 OpenAI 兼容 tools Schema（避免两套定义）。"""
    properties = {
        name: {"type": "string", "description": description}
        for name, description in tool.parameters.items()
    }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(tool.parameters),
            },
        },
    }


def _convert_message(message: Message) -> dict[str, object]:
    """把内部 Message 转换为 OpenAI 兼容消息（tool_calls / tool_call_id 语义与官方一致）。"""
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ],
        }
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id or "",
        }
    return {"role": message.role, "content": message.content}


def _parse_tool_call(raw: object) -> ToolCall:
    """解析单个 tool_call：function.name 必填，arguments 必须是 JSON 对象字符串。"""
    if not isinstance(raw, dict):
        raise InvalidResponseError("tool_calls 元素不是对象")
    function = raw.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise InvalidResponseError("tool_call 缺少 function.name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(raw_arguments, str):
        raise InvalidResponseError("function.arguments 必须是 JSON 字符串")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise InvalidResponseError(f"function.arguments 不是合法 JSON：{exc}") from exc
    if not isinstance(arguments, dict):
        raise InvalidResponseError("function.arguments 必须是 JSON 对象")
    call_id = raw.get("id")
    return ToolCall(
        id=call_id if isinstance(call_id, str) else "",
        name=function["name"],
        arguments={str(key): str(value) for key, value in arguments.items()},
    )


def _parse_reply(data: dict[str, object]) -> ModelReply:
    """从 /chat/completions 响应解析 ModelReply；非法结构抛 InvalidResponseError。"""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidResponseError("响应缺少 choices[0].message 字段") from exc
    if not isinstance(message, dict):
        raise InvalidResponseError("choices[0].message 不是对象")
    raw_calls = message.get("tool_calls") or []
    if raw_calls:
        if not isinstance(raw_calls, list):
            raise InvalidResponseError("tool_calls 不是数组")
        return ModelReply(tool_calls=[_parse_tool_call(raw) for raw in raw_calls])
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return ModelReply(content=content)
    raise InvalidResponseError("响应既没有可用的 tool_calls，也没有 content")


class OpenAICompatibleAgentModelClient:
    """通过 OpenAI 兼容 function calling 接口驱动 Agent（已验证目标：智谱 GLM）。

    官方接口（2026-11 文档确认）：
    - 请求 tools：[{"type": "function", "function": {name, description, parameters}}]，
      tool_choice 默认且仅支持 auto；
    - 响应 message.tool_calls[]：{id, function: {name, arguments(JSON 字符串)}}；
    - 回传：assistant 消息（含 tool_calls）+ {"role": "tool", "content", "tool_call_id"}。
    """

    provider = "openai-compatible"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """保存配置；transport 仅供测试注入（httpx.MockTransport），生产不传。"""
        self._settings = settings
        self._transport = transport

    def chat(self, messages: list[Message], tools: Sequence[Tool]) -> ModelReply:
        """发送对话与工具定义，解析出最终回答或工具调用请求。"""
        assert self._settings.api_key is not None  # load_settings 已保证非空
        url = (self._settings.base_url or "").rstrip("/") + "/chat/completions"
        payload: dict[str, object] = {
            "model": self._settings.generation_model,
            "messages": [_convert_message(message) for message in messages],
        }
        if tools:
            payload["tools"] = [_tool_to_schema(tool) for tool in tools]
        data = post_json(
            url,
            api_key=self._settings.api_key,
            payload=payload,
            timeout_seconds=self._settings.timeout_seconds,
            transport=self._transport,
        )
        return _parse_reply(data)
