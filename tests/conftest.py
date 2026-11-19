"""Pytest 公共夹具：隔离的数据目录与真实应用实例。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_agent_workflow.config import load_settings
from ai_agent_workflow.main import create_app


@pytest.fixture
def settings(tmp_path):
    """测试配置：隔离数据目录、不读本机 .env、静音日志。"""
    return load_settings(
        {
            "DATA_DIR": str(tmp_path / "data"),
            "LOG_LEVEL": "WARNING",
            "ENVIRONMENT": "test",
        }
    )


@pytest.fixture
def client(settings):
    """带 lifespan 的 HTTP 客户端（启动时自动执行迁移）。"""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
