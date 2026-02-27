"""
Defines AppContext, class holding global state and configuration - db instances, cache connections, settings

AppContext is used as lifespan in FastAPI so it supports any required lifecycle operations during startup and shutdown
of the server.
"""

from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Type, Any

from fastapi import FastAPI

from parkupine.settings import AppSettings


class AppContext(AbstractAsyncContextManager[dict[str, Any]]):
    app: FastAPI
    settings: AppSettings

    def __init__(self, app: FastAPI):
        self.app = app
        self.app.context = self  # type: ignore[attr-defined]
        self.settings = AppSettings()

    async def __aenter__(self) -> dict[str, Any]:
        """Lifespan setup. Can return dict that becomes starlette's scope["state"]"""
        return {}

    async def __aexit__(
        self, exc_type: Type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Lifespan teardown"""
        pass
