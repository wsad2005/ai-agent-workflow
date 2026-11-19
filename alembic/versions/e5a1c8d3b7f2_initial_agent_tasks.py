"""初始版本：agent_tasks 表（输入/状态/计划/结果/确认记录）。

Revision ID: e5a1c8d3b7f2
Revises:
Create Date: 2026-11-19
"""


import sqlalchemy as sa

from alembic import op

revision: str = "e5a1c8d3b7f2"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "agent_tasks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("input", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="pending"),
        sa.Column("planner", sa.String(length=16), nullable=False, server_default="rule"),
        sa.Column("plan", sa.JSON(), nullable=True),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("current_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("observations", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("approved_steps", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_agent_tasks"),
        sa.CheckConstraint(
            "status IN ('pending','planning','waiting_confirmation','running',"
            "'succeeded','failed','cancelled')",
            name="ck_agent_tasks_status",
        ),
    )
    op.create_index("ix_agent_tasks_status", "agent_tasks", ["status"])
    op.create_index("ix_agent_tasks_created_at", "agent_tasks", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_tasks_created_at", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_status", table_name="agent_tasks")
    op.drop_table("agent_tasks")
