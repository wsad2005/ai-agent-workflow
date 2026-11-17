"""异常分类与用户友好提示（与 01/02 项目同一风格）。

约定：所有异常消息都不包含 API Key 等秘密信息，可以安全进入日志。
"""


class AgentError(Exception):
    """本项目所有自定义异常的基类。"""


class ConfigError(AgentError):
    """配置缺失或取值非法（环境变量 / .env）。"""


class EmptyInputError(AgentError):
    """用户输入为空或全部是空白。"""


class ToolNotFoundError(AgentError):
    """请求的工具未在注册表中。"""


class ToolExecutionError(AgentError):
    """工具执行失败（含表达式无法计算等情况）。"""


class ModelError(AgentError):
    """模型客户端返回了不符合约定的结果。"""


class AgentRequestError(AgentError):
    """调用模型服务过程中的失败（超时、连接、认证、限流、服务端错误、无效响应）。"""


class AgentTimeoutError(AgentRequestError):
    """请求在限定时间内未得到响应。"""


class AgentConnectionError(AgentRequestError):
    """无法连接到服务地址。"""


class AgentAuthError(AgentRequestError):
    """API Key 无效或缺少权限。"""


class AgentRateLimitError(AgentRequestError):
    """触发服务商限流（HTTP 429）。"""


class AgentServerError(AgentRequestError):
    """模型服务端内部错误（HTTP 5xx）。"""


class InvalidResponseError(AgentRequestError):
    """服务返回的内容无法解析或不符合约定。"""


_FRIENDLY_MESSAGES: dict[type[Exception], str] = {
    ConfigError: "配置有误：请检查 .env 与环境变量，参考 README 的配置变量说明。",
    EmptyInputError: "输入为空：请先输入你的问题。",
    ToolNotFoundError: "请求的工具不存在。",
    ToolExecutionError: "工具执行失败，请检查输入后重试。",
    ModelError: "模型返回了无法处理的结果，请稍后重试。",
    AgentTimeoutError: "请求超时：服务未在限定时间内响应，可稍后重试或调大 LLM_TIMEOUT_SECONDS。",
    AgentConnectionError: "连接失败：无法访问服务地址，请检查网络与 LLM_BASE_URL。",
    AgentAuthError: (
        "认证失败：API Key 无效或权限不足，请在本地 .env 中核对（密钥不要提供给任何人）。"
    ),
    AgentRateLimitError: "触发限流：请求过于频繁，请稍后重试。",
    AgentServerError: "服务端错误：模型服务暂时不可用，请稍后重试。",
    InvalidResponseError: "服务返回了无法解析的内容，请稍后重试或查看日志。",
    AgentRequestError: "调用模型服务失败，请稍后重试。",
    AgentError: "程序出现问题，请查看日志获取详情。",
}


def friendly_message(exc: Exception) -> str:
    """按异常类型（沿 MRO 从具体到一般）返回面向用户的安全提示。"""
    for exc_type in type(exc).__mro__:
        message = _FRIENDLY_MESSAGES.get(exc_type)
        if message is not None:
            if isinstance(exc, ConfigError):
                return f"{message}详细信息：{exc}"
            return message
    return "发生意外错误，请查看日志获取详情。"
