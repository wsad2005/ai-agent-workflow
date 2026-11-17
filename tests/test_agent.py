"""agent 主循环测试：端到端、安全阀、异常路径与消息协议（全离线）。"""

import pytest

from ai_agent_workflow.agent import Agent
from ai_agent_workflow.errors import EmptyInputError, ModelError
from ai_agent_workflow.model import Message, MockAgentModelClient, ModelReply, ToolCall
from ai_agent_workflow.tools import ToolRegistry, default_registry


def test_end_to_end_calculator() -> None:
    agent = Agent(model=MockAgentModelClient(), registry=default_registry())
    assert "14" in agent.run("计算 2+3*4 等于多少？")


def test_end_to_end_time_tool() -> None:
    agent = Agent(model=MockAgentModelClient(), registry=default_registry())
    answer = agent.run("现在几点了？")
    assert "[mock] 工具结果：" in answer  # 时间字符串经工具回灌进最终回答


def test_unknown_tool_error_fed_back_and_answered() -> None:
    agent = Agent(model=MockAgentModelClient(), registry=ToolRegistry())  # 空注册表
    answer = agent.run("计算 2+3 等于多少？")
    assert "工具执行失败" in answer  # 错误作为观察回灌，循环不崩溃


def test_empty_input_raises() -> None:
    agent = Agent(model=MockAgentModelClient(), registry=default_registry())
    with pytest.raises(EmptyInputError):
        agent.run("   ")


def test_loop_cap_forces_stop() -> None:
    class AlwaysToolCallClient:
        def chat(self, messages: list[Message], tools: object) -> ModelReply:
            return ModelReply(
                tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+1"})]
            )

    agent = Agent(
        model=AlwaysToolCallClient(),  # type: ignore[arg-type]  # 测试桩：永远请求工具
        registry=default_registry(),
        max_tool_rounds=2,
    )
    assert "工具调用上限" in agent.run("随便算点什么")


def test_malformed_model_reply_raises() -> None:
    class EmptyReplyClient:
        def chat(self, messages: list[Message], tools: object) -> ModelReply:
            return ModelReply()

    agent = Agent(
        model=EmptyReplyClient(),  # type: ignore[arg-type]  # 测试桩：既不回答也不调用
        registry=default_registry(),
    )
    with pytest.raises(ModelError):
        agent.run("hi")


def test_invalid_max_tool_rounds_rejected() -> None:
    with pytest.raises(ValueError):
        Agent(model=MockAgentModelClient(), registry=default_registry(), max_tool_rounds=0)


def test_tool_call_id_flows_into_conversation() -> None:
    """真实 API 的回传语义：assistant 消息携带 tool_calls，tool 消息携带 tool_call_id。"""
    captured: dict[str, list[Message]] = {}

    class TwoRoundClient:
        def chat(self, messages: list[Message], tools: object) -> ModelReply:
            captured["latest"] = messages
            if any(message.role == "tool" for message in messages):
                return ModelReply(content="[stub] 完成")
            return ModelReply(
                tool_calls=[
                    ToolCall(id="call_abc", name="calculator", arguments={"expression": "1+1"})
                ]
            )

    agent = Agent(model=TwoRoundClient(), registry=default_registry())  # type: ignore[arg-type]
    assert "[stub] 完成" == agent.run("计算 1+1")
    second_round = captured["latest"]
    assistant = next(m for m in second_round if m.role == "assistant")
    tool_message = next(m for m in second_round if m.role == "tool")
    assert assistant.tool_calls[0].id == "call_abc"
    assert tool_message.tool_call_id == "call_abc"
    assert tool_message.content == "2"  # 计算器真实执行结果
