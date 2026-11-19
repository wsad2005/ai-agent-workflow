"""工具注册表接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ai_agent_workflow.api.deps import Container, get_container
from ai_agent_workflow.api.schemas import ToolOut

router = APIRouter(tags=["工具"])


@router.get(
    "",
    response_model=list[ToolOut],
    summary="列出全部注册工具（含权限级别与超时）",
)
def list_tools(container: Container = Depends(get_container)) -> list[ToolOut]:
    """返回注册表中的全部工具及权限信息。"""
    return [
        ToolOut(
            name=tool.name,
            description=tool.description,
            parameters=dict(tool.parameters),
            permission=tool.permission,
            timeout_seconds=tool.timeout_seconds,
        )
        for tool in container.registry.tools()
    ]
