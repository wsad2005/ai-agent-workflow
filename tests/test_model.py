"""model 模块测试：Mock 决策策略与 OpenAI 兼容真实客户端（HTTP 全部 Mock）。"""

import json
from collections.abc import Callable

import httpx
import pytest

from ai_agent_workflow.config import load_settings
from ai_agent_workflow.errors import (
    AgentAuthError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentServerError,
    AgentTimeoutError,
    InvalidResponseError,
)
from ai_agent_workflow.model import (
    AgentModelClient,
    Message,
    MockAgentModelClient,
    OpenAICompatibleAgentModelClient,
)
from ai_agent_workflow.tools import default_registry

BASE_ENV = {
    "AGENT_MODE": "real",
    "LLM_BASE_URL": "https://api.example.invalid/v1",  # 测试用假地址
    "LLM_API_KEY": "dummy-key-abcdef123456",  # 测试用假密钥
    "GENERATION_MODEL": "dummy-generation",  # 测试用假模型名
}


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenAICompatibleAgentModelClient:
    settings = load_settings(environ=dict(BASE_ENV))
    return OpenAICompatibleAgentModelClient(settings, transport=httpx.MockTransport(handler))


def test_mock_satisfies_protocol() -> None:
    assert isinstance(MockAgentModelClient(), AgentModelClient)


def test_math_question_triggers_calculator_call() -> None:
    reply = MockAgentModelClient().chat(
        [Message(role="user", content="计算 2+3*4 等于多少？")], tools=[]
    )
    assert reply.content is None
    assert reply.tool_calls[0].name == "calculator"
    assert reply.tool_calls[0].arguments["expression"] == "2+3*4"


def test_time_question_triggers_time_call() -> None:
    reply = MockAgentModelClient().chat([Message(role="user", content="现在几点了？")], tools=[])
    assert reply.tool_calls[0].name == "current_time"


def test_tool_observation_produces_final_answer() -> None:
    messages = [
        Message(role="user", content="计算 1+1"),
        Message(role="assistant", content="调用工具 calculator"),
        Message(role="tool", content="2", tool_name="calculator"),
    ]
    reply = MockAgentModelClient().chat(messages, tools=[])
    assert reply.content is not None
    assert "2" in reply.content
    assert not reply.tool_calls


def test_plain_question_answered_directly() -> None:
    reply = MockAgentModelClient().chat([Message(role="user", content="你好")], tools=[])
    assert reply.content is not None
    assert not reply.tool_calls


def test_invalid_max_tool_rounds_rejected() -> None:
    with pytest.raises(ValueError):
        MockAgentModelClient(max_tool_rounds=0)


def test_real_client_sends_tool_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        assert request.url.path.endswith("/chat/completions")
        assert "dummy-key-abcdef123456" not in str(request.url)  # 密钥只在请求头
        assert request.headers["Authorization"] == "Bearer dummy-key-abcdef123456"
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "好的"}}]}
        )

    reply = make_client(handler).chat(
        [Message(role="user", content="你好")], tools=default_registry().tools()
    )
    assert reply.content == "好的"
    body = captured["body"]
    assert body["model"] == "dummy-generation"
    tools = body["tools"]
    assert isinstance(tools, list)
    calculator = next(tool for tool in tools if tool["function"]["name"] == "calculator")
    assert calculator["type"] == "function"
    parameters = calculator["function"]["parameters"]
    assert parameters["required"] == ["expression"]
    assert parameters["properties"]["expression"]["type"] == "string"


def test_real_client_parses_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression": "2+3*4"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    reply = make_client(handler).chat([Message(role="user", content="计算 2+3*4")], tools=[])
    assert reply.content is None
    assert reply.tool_calls[0].name == "calculator"
    assert reply.tool_calls[0].arguments == {"expression": "2+3*4"}
    assert reply.tool_calls[0].id == "call_1"


def test_real_client_parses_final_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "最终回答"}}]},
        )

    reply = make_client(handler).chat([Message(role="user", content="你好")], tools=[])
    assert reply.content == "最终回答"
    assert not reply.tool_calls


def test_real_client_round_trips_tool_messages() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("bodies", []).append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "完成",
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression": "1+1"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = make_client(handler)
    first = client.chat([Message(role="user", content="计算 1+1")], tools=[])
    # 模拟 Agent 主循环的回灌：assistant 消息带 tool_calls，tool 消息带 tool_call_id
    client.chat(
        [
            Message(role="user", content="计算 1+1"),
            Message(role="assistant", content="", tool_calls=list(first.tool_calls)),
            Message(role="tool", content="2", tool_name="calculator", tool_call_id="call_9"),
        ],
        tools=[],
    )
    bodies = captured["bodies"]
    second_messages = bodies[1]["messages"]
    assistant = second_messages[-2]
    tool_message = second_messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_9"
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "expression": "1+1"
    }
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_9"
    assert tool_message["content"] == "2"


def test_real_client_invalid_arguments_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "calculator", "arguments": "not-json"},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    with pytest.raises(InvalidResponseError):
        make_client(handler).chat([Message(role="user", content="q")], tools=[])


def test_real_client_missing_function_name() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{"id": "call_1", "type": "function", "function": {}}],
                        }
                    }
                ]
            },
        )

    with pytest.raises(InvalidResponseError):
        make_client(handler).chat([Message(role="user", content="q")], tools=[])


def test_real_client_empty_reply_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}]})

    with pytest.raises(InvalidResponseError):
        make_client(handler).chat([Message(role="user", content="q")], tools=[])


@pytest.mark.parametrize(
    ("status", "exc_type"),
    [
        (401, AgentAuthError),
        (429, AgentRateLimitError),
        (500, AgentServerError),
        (400, InvalidResponseError),
    ],
)
def test_real_client_http_errors_map(status: int, exc_type: type[Exception]) -> None:
    client = make_client(lambda request: httpx.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(exc_type):
        client.chat([Message(role="user", content="q")], tools=[])


def test_real_client_auth_error_message_never_contains_api_key() -> None:
    client = make_client(lambda request: httpx.Response(401, json={"error": {"message": "x"}}))
    with pytest.raises(AgentAuthError) as excinfo:
        client.chat([Message(role="user", content="q")], tools=[])
    assert BASE_ENV["LLM_API_KEY"] not in str(excinfo.value)


def test_real_client_timeout_maps_to_agent_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(AgentTimeoutError):
        make_client(handler).chat([Message(role="user", content="q")], tools=[])


def test_real_client_connection_error_maps_to_agent_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(AgentConnectionError):
        make_client(handler).chat([Message(role="user", content="q")], tools=[])
