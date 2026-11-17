"""组件工厂：按 Settings.mode 创建 Mock 或真实模型客户端。"""

from .config import MOCK_MODE, REAL_MODE, Settings
from .errors import ConfigError
from .model import AgentModelClient, MockAgentModelClient, OpenAICompatibleAgentModelClient

__all__ = ["create_model_client"]


def create_model_client(settings: Settings) -> AgentModelClient:
    """按配置创建模型客户端。"""
    if settings.mode == MOCK_MODE:
        return MockAgentModelClient()
    if settings.mode == REAL_MODE:
        return OpenAICompatibleAgentModelClient(settings)
    raise ConfigError(f"未知的 AGENT_MODE：{settings.mode!r}")
