"""errors 模块测试。"""

import pytest

from ai_agent_workflow.errors import (
    AgentAuthError,
    AgentConnectionError,
    AgentError,
    AgentRateLimitError,
    AgentRequestError,
    AgentServerError,
    AgentTimeoutError,
    ConfigError,
    EmptyInputError,
    InvalidResponseError,
    ModelError,
    ToolExecutionError,
    ToolNotFoundError,
    friendly_message,
)

REQUEST_ERRORS = [
    AgentTimeoutError,
    AgentConnectionError,
    AgentAuthError,
    AgentRateLimitError,
    AgentServerError,
    InvalidResponseError,
]


@pytest.mark.parametrize("exc_type", REQUEST_ERRORS)
def test_request_errors_share_base(exc_type: type[AgentRequestError]) -> None:
    assert issubclass(exc_type, AgentRequestError)
    assert issubclass(exc_type, AgentError)


@pytest.mark.parametrize(
    ("exc", "keyword"),
    [
        (AgentTimeoutError("t"), "超时"),
        (AgentConnectionError("c"), "连接"),
        (AgentAuthError("a"), "认证"),
        (AgentRateLimitError("r"), "限流"),
        (AgentServerError("s"), "服务端错误"),
        (InvalidResponseError("i"), "解析"),
        (ConfigError("x"), "配置"),
        (EmptyInputError("e"), "输入为空"),
        (ToolNotFoundError("n"), "工具"),
        (ToolExecutionError("x"), "工具执行失败"),
        (ModelError("m"), "模型"),
    ],
)
def test_friendly_message_by_type(exc: Exception, keyword: str) -> None:
    assert keyword in friendly_message(exc)


def test_config_error_message_includes_detail() -> None:
    assert "LLM_API_KEY 缺少" in friendly_message(ConfigError("LLM_API_KEY 缺少"))


def test_friendly_message_falls_back_for_unknown() -> None:
    assert friendly_message(ValueError("boom")) == "发生意外错误，请查看日志获取详情。"
