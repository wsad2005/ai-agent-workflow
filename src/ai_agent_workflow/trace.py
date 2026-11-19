"""JSONL 执行追踪：一次一行；task_id 区分任务；敏感值脱敏。

脱敏规则：参数或字段名包含 key/token/secret/password/authorization 时，
值替换为 ``***``——API Key、密码、Token 永远不会进入日志文件。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

_SECRET_KEY_HINTS = ("key", "token", "secret", "password", "authorization")


def redact_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """按键名提示词脱敏；其余原样保留。"""
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = str(key).lower()
        result[key] = "***" if any(hint in lowered for hint in _SECRET_KEY_HINTS) else value
    return result


class TraceRecorder:
    """把执行事件追加写入 JSONL 文件；支持按 task_id 过滤读取。"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """追踪文件路径。"""
        return self._path

    def append(
        self,
        *,
        task_id: str,
        trace_id: str,
        action: str,
        tool_name: str = "",
        status: str = "",
        duration_ms: int | None = None,
        error: str | None = None,
        arguments: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """追加一条追踪记录（自动脱敏参数）。"""
        record: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "task_id": task_id,
            "trace_id": trace_id,
            "action": action,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": duration_ms,
            "error": error,
            "arguments": redact_mapping(arguments or {}),
        }
        record.update(extra)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """读取全部记录；文件不存在返回空列表。"""
        if not self._path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def read_by_task(self, task_id: str) -> list[dict[str, Any]]:
        """读取指定任务的追踪记录。"""
        return [record for record in self.read_all() if record.get("task_id") == task_id]
