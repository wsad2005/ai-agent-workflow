"""`python -m ai_agent_workflow` 演示：Mock 模式离线免费；real 模式走真实模型。"""

import sys

from .agent import Agent
from .config import load_settings
from .errors import AgentError, friendly_message
from .factory import create_model_client
from .logging_setup import configure_logging
from .tools import default_registry

QUESTIONS = (
    "计算 2+3*4 等于多少？",
    "现在几点了？",
    "用一句话介绍什么是 AI Agent",
)


def main() -> None:
    """按当前配置跑三个示例问题；失败时输出友好提示并以非零码退出。"""
    configure_logging()
    try:
        settings = load_settings()
        agent = Agent(model=create_model_client(settings), registry=default_registry())
        print(f"[demo] 配置：{settings.describe()}")
        for question in QUESTIONS:
            print(f"[demo] 问题：{question}")
            print(f"[demo] 回答：{agent.run(question)}")
            print()
    except AgentError as exc:
        print(friendly_message(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
