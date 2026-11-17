"""config 模块测试（全部使用假值）。"""

import pytest

from ai_agent_workflow.config import (
    DEFAULT_TIMEOUT_SECONDS,
    Settings,
    load_settings,
    mask_secret,
)
from ai_agent_workflow.errors import ConfigError

REAL_ENV = {
    "AGENT_MODE": "real",
    "LLM_BASE_URL": "https://api.example.invalid/v1",  # 测试用假地址
    "LLM_API_KEY": "dummy-key-abcdef123456",  # 测试用假密钥
    "GENERATION_MODEL": "dummy-generation",  # 测试用假模型名
}


def test_default_is_mock_without_any_config() -> None:
    settings = load_settings(environ={})
    assert settings.mode == "mock"
    assert settings.base_url is None
    assert settings.api_key is None
    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_mode_is_case_insensitive() -> None:
    assert load_settings(environ={"AGENT_MODE": " MOCK "}).mode == "mock"
    assert load_settings(environ={**REAL_ENV, "AGENT_MODE": " Real "}).mode == "real"


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings(environ={"AGENT_MODE": "hybrid"})
    assert "hybrid" in str(excinfo.value)


def test_real_mode_without_config_lists_missing_vars() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings(environ={"AGENT_MODE": "real"})
    message = str(excinfo.value)
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "GENERATION_MODEL"):
        assert name in message


def test_real_mode_settings_built_when_config_complete() -> None:
    settings = load_settings(environ={**REAL_ENV, "LLM_TIMEOUT_SECONDS": "7"})
    assert settings == Settings(
        mode="real",
        base_url="https://api.example.invalid/v1",
        api_key="dummy-key-abcdef123456",
        generation_model="dummy-generation",
        timeout_seconds=7,
    )


@pytest.mark.parametrize("raw", ["abc", "0", "-3", "1.5"])
def test_invalid_timeout_raises_config_error(raw: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(environ={"LLM_TIMEOUT_SECONDS": raw})


def test_blank_values_treated_as_unset() -> None:
    settings = load_settings(environ={"LLM_API_KEY": "", "GENERATION_MODEL": "  "})
    assert settings.api_key is None
    assert settings.generation_model is None


def test_mask_secret_short_value_fully_hidden() -> None:
    assert mask_secret("abc") == "***"


def test_mask_secret_long_value_keeps_edges_only() -> None:
    assert mask_secret("supersecret123456") == "su***56"


def test_mask_secret_none() -> None:
    assert mask_secret(None) == "未设置"


def test_describe_never_contains_plain_api_key() -> None:
    env = {**REAL_ENV, "LLM_API_KEY": "supersecret123456"}
    summary = load_settings(environ=env).describe()
    assert "supersecret123456" not in summary
    assert "su***56" in summary
