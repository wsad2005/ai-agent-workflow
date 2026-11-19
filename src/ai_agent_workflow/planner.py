"""规划器：Rule-based Planner（默认、离线）与可选 LLMPlanner 的统一接口。

- RuleBasedPlanner：把 v0.2 Mock 模型的规则决策正式化为规划器（算式→calculator；
  时间→current_time；其余直接回答）；
- LLMPlanner：通过 OpenAI 兼容接口生成计划，输出必须通过工具白名单与参数
  Schema 校验（未知工具/未知参数一律 PlanningError），绝不直接执行模型文本。

环境变量 PLANNER_MODE=rule（默认）| llm 控制选择；未配置 LLM 时自动回退规则式。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import PlanningError
from .logging_setup import get_logger
from .tools import Tool

logger = get_logger(__name__)

_MATH_CANDIDATE = re.compile(r"[-+*/()0-9.\s]{3,}")
_HAS_DIGIT = re.compile(r"\d")
_HAS_OPERATOR = re.compile(r"[-+*/]")
_TIME_KEYWORDS = ("时间", "几点", "日期")

_MAX_LLM_STEPS = 10


def extract_expression(text: str) -> str | None:
    """从文本中提取首个「像算式」的片段；没有则返回 None。"""
    for candidate in _MATH_CANDIDATE.findall(text):
        cleaned = candidate.strip()
        if _HAS_DIGIT.search(cleaned) and _HAS_OPERATOR.search(cleaned):
            return cleaned
    return None


@dataclass(frozen=True)
class PlanStep:
    """计划中的一个工具调用步骤。"""

    tool_name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class AgentPlan:
    """规划结果：若干工具步骤 + 可选的直接回答（无工具步骤时）。"""

    steps: tuple[PlanStep, ...] = ()
    direct_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """可持久化的 JSON 结构（存入任务表 plan 字段）。"""
        return {
            "steps": [
                {"tool_name": step.tool_name, "arguments": dict(step.arguments)}
                for step in self.steps
            ],
            "direct_answer": self.direct_answer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentPlan:
        """从持久化结构恢复计划。"""
        steps = tuple(
            PlanStep(
                tool_name=str(entry["tool_name"]),
                arguments={str(k): str(v) for k, v in entry.get("arguments", {}).items()},
            )
            for entry in data.get("steps", [])
        )
        direct = data.get("direct_answer")
        return cls(steps=steps, direct_answer=str(direct) if direct else None)


@runtime_checkable
class Planner(Protocol):
    """规划器接口：目标 + 可用工具 → 计划。"""

    provider: str

    def plan(self, goal: str, tools: Sequence[Tool]) -> AgentPlan:
        """产出执行计划。"""
        ...


class RuleBasedPlanner:
    """规则式规划器（默认）：确定性、离线、零费用。"""

    provider = "rule"

    def plan(self, goal: str, tools: Sequence[Tool]) -> AgentPlan:
        """按规则产生计划；仅使用注册表中真实存在的工具。"""
        known = {tool.name for tool in tools}
        expression = extract_expression(goal)
        if expression is not None and "calculator" in known:
            return AgentPlan(
                steps=(PlanStep(tool_name="calculator", arguments={"expression": expression}),)
            )
        if any(keyword in goal for keyword in _TIME_KEYWORDS) and "current_time" in known:
            return AgentPlan(steps=(PlanStep(tool_name="current_time", arguments={}),))
        return AgentPlan(direct_answer=f"[agent] 已收到你的问题：{goal}")


class LLMPlanner:
    """LLM 规划器（需配置 ai_base_url / ai_api_key / ai_model）。

    输出必须为纯 JSON：{"steps": [{"tool_name", "arguments"}], "direct_answer": str|null}。
    服务端逐项校验工具白名单与参数 Schema；任何越界都抛 PlanningError。
    """

    provider = "llm"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    def plan(self, goal: str, tools: Sequence[Tool]) -> AgentPlan:
        """调用 LLM 生成计划并做白名单/Schema 校验。"""
        tool_catalog = [
            {
                "tool_name": tool.name,
                "description": tool.description,
                "parameters": sorted(tool.parameters),
            }
            for tool in tools
        ]
        system = (
            "你是 Agent 的任务规划器。只输出纯 JSON（不要 markdown 代码块、不要解释），结构："
            '{"steps": [{"tool_name": "工具名", "arguments": {"参数名": "字符串值"}}], '
            '"direct_answer": "无需工具时的一句话回答，否则为 null"}'
            "。只能使用工具列表中的工具名与参数名； arguments 的值必须是字符串。使用简体中文。"
        )
        user = f"目标：{goal}\n可用工具：{json.dumps(tool_catalog, ensure_ascii=False)}"
        try:
            # transport 仅 httpx.Client 支持；模块级 httpx.post 没有 transport 参数
            with httpx.Client(
                timeout=self._settings.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(
                    (self._settings.base_url or "").rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self._settings.api_key or ''}"},
                    json={
                        "model": self._settings.generation_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.1,
                    },
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise PlanningError(f"LLM 规划请求失败：{exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise PlanningError("LLM 响应结构无法解析") from exc

        return self._validate_plan(content, tools)

    def _validate_plan(self, content: str, tools: Sequence[Tool]) -> AgentPlan:
        """解析并校验 LLM 计划：工具白名单 + 参数 Schema + 步数上限。"""
        text = content.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        try:
            raw: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanningError("LLM 输出不是合法 JSON") from exc

        known = {tool.name: tool for tool in tools}
        steps: list[PlanStep] = []
        for entry in raw.get("steps", [])[:_MAX_LLM_STEPS]:
            tool_name = str(entry.get("tool_name", ""))
            tool = known.get(tool_name)
            if tool is None:
                raise PlanningError(f"模型请求了未知工具：{tool_name}")
            arguments = entry.get("arguments", {})
            if not isinstance(arguments, dict):
                raise PlanningError(f"工具 {tool_name} 的 arguments 不是对象")
            unknown = sorted(set(arguments) - set(tool.parameters))
            if unknown:
                raise PlanningError(
                    f"工具 {tool_name} 的参数不符合 Schema：未知参数 {', '.join(unknown)}"
                )
            missing = sorted(set(tool.parameters) - set(arguments))
            if missing:
                raise PlanningError(
                    f"工具 {tool_name} 的参数不符合 Schema：缺少 {', '.join(missing)}"
                )
            steps.append(
                PlanStep(
                    tool_name=tool_name,
                    arguments={str(k): str(v) for k, v in arguments.items()},
                )
            )
        direct_answer = raw.get("direct_answer")
        if not steps and not (isinstance(direct_answer, str) and direct_answer.strip()):
            raise PlanningError("LLM 计划既没有工具步骤也没有直接回答")
        logger.info("LLM 规划完成：%d 个工具步骤", len(steps))
        return AgentPlan(
            steps=tuple(steps),
            direct_answer=(
                str(direct_answer)
                if isinstance(direct_answer, str) and direct_answer.strip()
                else None
            ),
        )


def build_planner(
    settings: Settings, transport: httpx.BaseTransport | None = None
) -> Planner:
    """按配置构建规划器：PLANNER_MODE=llm 且 AI 配置齐全时用 LLM，否则规则式。"""
    if settings.planner_mode == "llm" and settings.ai_configured:
        return LLMPlanner(settings, transport=transport)
    return RuleBasedPlanner()
