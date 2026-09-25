import logging

import httpx
import openai
import psycopg
from fastapi.responses import JSONResponse
from ollama import ResponseError

from faq_service.domain.errors import ServiceError


def register_handlers(app):
    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code, content={"code": exc.code, "message": exc.message}
        )

    @app.exception_handler(TimeoutError)
    @app.exception_handler(openai.APITimeoutError)
    @app.exception_handler(httpx.TimeoutException)
    async def timeout_error(request, exc):
        return JSONResponse(
            status_code=504,
            content={
                "code": "timeout",
                "message": "Время ожидания ответа истекло.",
            },
        )

    @app.exception_handler(psycopg.Error)
    @app.exception_handler(openai.APIError)
    @app.exception_handler(httpx.RequestError)
    @app.exception_handler(ResponseError)
    @app.exception_handler(ConnectionError)
    async def dependency_error(request, exc):
        logging.getLogger(__name__).error("Dependency error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "code": "dependency_unavailable",
                "message": "Сервис временно недоступен.",
            },
        )

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logging.getLogger(__name__).error("Request error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "Не удалось обработать запрос.",
            },
        )
