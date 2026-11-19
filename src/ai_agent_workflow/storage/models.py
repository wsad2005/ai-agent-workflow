"""Agent 任务 ORM 模型与声明式基类。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """全部 ORM 模型的声明式基类。"""


def now_local() -> datetime:
    """本地时区 aware 时间（趋势与展示以用户日历为准）。"""
    from datetime import datetime as dt

    return dt.now().astimezone()


class AgentTask(Base):
    """Agent 任务：输入、计划、执行状态与结果的持久化载体。

    状态机（见 task_store.ALLOWED_TRANSITIONS）：
    pending → planning → running → succeeded/failed
    running → waiting_confirmation → running（批准恢复）/ failed（拒绝）
    pending/planning/waiting_confirmation/running → cancelled
    """

    __tablename__ = "agent_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    input: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending", index=True)
    planner: Mapped[str] = mapped_column(String(16), nullable=False, default="rule")
    plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    observations: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    approved_steps: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now_local
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=now_local,
        onupdate=now_local,
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<AgentTask id={self.id} status={self.status!r} input={self.input!r}>"
