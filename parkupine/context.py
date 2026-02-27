"""
Defines AppContext, class holding global state and configuration - db instances, cache connections, settings

AppContext is used as lifespan in FastAPI so it supports any required lifecycle operations during startup and shutdown
of the server.
"""
from fastapi import FastAPI

from parkupine.settings import AppSettings


class AppContext:
    """
    Also acts as a lifespan
    """
    app: FastAPI
    settings: AppSettings

    def __init__(self, app: FastAPI):
        self.app = app
        self.app.context = self
        self.settings = AppSettings()

    async def __aenter__(self):
        """Lifespan setup"""
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Lifespan teardown"""
        pass
