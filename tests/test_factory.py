"""factory 模块测试。"""

from ai_agent_workflow.config import load_settings
from ai_agent_workflow.factory import create_model_client
from ai_agent_workflow.model import MockAgentModelClient, OpenAICompatibleAgentModelClient

MOCK_ENV: dict[str, str] = {"AGENT_MODE": "mock"}

REAL_ENV = {
    "AGENT_MODE": "real",
    "LLM_BASE_URL": "https://api.example.invalid/v1",  # 测试用假地址
    "LLM_API_KEY": "dummy-key-abcdef123456",  # 测试用假密钥
    "GENERATION_MODEL": "dummy-generation",  # 测试用假模型名
}


def test_mock_mode_creates_mock_client() -> None:
    settings = load_settings(environ=dict(MOCK_ENV))
    assert isinstance(create_model_client(settings), MockAgentModelClient)


def test_real_mode_creates_openai_compatible_client() -> None:
    settings = load_settings(environ=dict(REAL_ENV))
    assert isinstance(create_model_client(settings), OpenAICompatibleAgentModelClient)
