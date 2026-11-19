"""统一异常处理：把各类异常收敛成 ``{"error": {...}}`` 响应。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_agent_workflow.errors import AgentError
from ai_agent_workflow.logging_setup import get_logger

logger = get_logger(__name__)


def _payload(code: str, message: str, detail: object = None) -> dict[str, object]:
    body: dict[str, object] = {"code": code, "message": message}
    if detail is not None:
        body["detail"] = detail
    return {"error": body}


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器。"""

    @app.exception_handler(AgentError)
    async def handle_app_error(request: Request, exc: AgentError) -> JSONResponse:
        logger.info(
            "业务异常 %s %s -> %s(%s): %s",
            request.method,
            request.url.path,
            getattr(exc, "code", "agent_error"),
            getattr(exc, "status_code", 400),
            str(exc),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(
            getattr(exc, 'code', 'agent_error'),
            str(exc),
            getattr(exc, 'detail', None),
        ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        simplified = [
            {"loc": item.get("loc"), "msg": item.get("msg"), "type": item.get("type")}
            for item in jsonable_encoder(exc.errors(), custom_encoder={})
        ]
        return JSONResponse(
            status_code=422,
            content=_payload("request_validation_error", "请求参数不合法", simplified),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(f"http_{exc.status_code}", str(exc.detail)),
        )
