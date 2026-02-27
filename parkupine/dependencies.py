"""
FastApi dependency injection nonsense
"""

from typing import Annotated, cast

from fastapi import Request
from fastapi.params import Depends

from parkupine.context import AppContext
from parkupine.settings import AppSettings


def get_context(request: Request) -> AppContext:
    """
    Dependency function that provides AppContext instance, as a shortcut
    """
    return cast(AppContext, request.app.context)


AppContextDep = Annotated[AppContext, Depends(get_context)]


def get_settings(context: AppContextDep) -> AppSettings:
    """
    Dependency function that provides AppSettings instance, as a shortcut
    """
    return context.settings


AppSettingsDep = Annotated[AppSettings, Depends(get_settings)]
