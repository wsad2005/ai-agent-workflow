"""Agent 主循环：模型决策 → 工具执行 → 结果回灌 → 最终回答。

这是整个项目的「工作流」核心：模型每轮要么给出最终回答，要么请求调用工具；
工具观察结果回灌进对话，循环直到给出回答或达到安全阀轮数。
assistant/tool 消息携带 tool_calls / tool_call_id（与 OpenAI 兼容回传语义一致），
主循环本身不感知任何具体服务商。
"""

import logging
from dataclasses import dataclass

from .errors import EmptyInputError, ModelError, ToolExecutionError, ToolNotFoundError
from .model import AgentModelClient, Message, ToolCall
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个严谨的中文智能体，可以调用工具回答问题。"
    "需要计算或查询时间时调用对应工具；拿到工具结果后再给出最终回答。"
)
DEFAULT_MAX_TOOL_ROUNDS = 3


@dataclass
class Agent:
    """最小 Agent：组合模型客户端与工具注册表。"""

    model: AgentModelClient
    registry: ToolRegistry
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS

    def __post_init__(self) -> None:
        if self.max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds 必须为正整数")

    def run(self, question: str) -> str:
        """处理一次用户输入，返回最终回答。"""
        text = question.strip()
        if not text:
            raise EmptyInputError("用户输入为空")
        messages: list[Message] = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=text),
        ]
        for _round in range(self.max_tool_rounds):
            reply = self.model.chat(messages, self.registry.tools())
            if reply.content is not None:
                logger.info("模型给出最终回答（长度 %d）", len(reply.content))
                return reply.content
            if not reply.tool_calls:
                raise ModelError("模型既没有给出回答，也没有请求调用工具")
            # 先记录模型本轮的全部调用请求（真实 API 需要 assistant+tool_calls 消息）
            messages.append(
                Message(role="assistant", content="", tool_calls=list(reply.tool_calls))
            )
            for call in reply.tool_calls:
                logger.info("模型请求调用工具 %s", call.name)
                observation = self._execute(call)
                messages.append(
                    Message(
                        role="tool",
                        content=observation,
                        tool_name=call.name,
                        tool_call_id=call.id or None,
                    )
                )
        logger.warning("达到工具调用上限 %d，强制结束", self.max_tool_rounds)
        return f"[agent] 已达到工具调用上限（{self.max_tool_rounds} 轮），已停止。"

    def _execute(self, call: ToolCall) -> str:
        """执行一次工具调用；失败信息作为观察结果回灌，而不是让循环崩溃。"""
        try:
            result = self.registry.execute(call.name, call.arguments)
        except ToolNotFoundError:
            logger.warning("工具不存在：%s", call.name)
            return f"工具执行失败：未注册的工具：{call.name}"
        except ToolExecutionError as exc:
            logger.warning("工具执行失败：%s", exc)
            return f"工具执行失败：{exc}"
        logger.info("工具 %s 执行成功（结果长度 %d）", call.name, len(result))
        return result
