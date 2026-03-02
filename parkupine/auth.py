"""
FastApi authentication dependencies and shenanigans
"""

from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from parkupine.dependencies import AppSettingsDep

bearer = HTTPBearer()


class BaseUser(BaseModel):
    """
    Model representing a User.
    """

    id: str
    name: str
    email: str
    role: Literal["user", "admin"]

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class OpenwebuiUserHeaders(BaseModel):
    """
    Defines headers expected to be sent by OpenWebui.
    These are used for user identification.

    Reference: https://docs.openwebui.com/reference/env-configuration/#enable_forward_user_info_headers
    """

    x_openwebui_user_name: str = Field(alias="x-openwebui-user-name")
    x_openwebui_user_email: str = Field(alias="x-openwebui-user-email")
    x_openwebui_user_id: str = Field(alias="x-openwebui-user-id")


def user_required(
    settings: AppSettingsDep,
    openwebui_user_headers: Annotated[OpenwebuiUserHeaders, Header()],
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> BaseUser:
    """
    If bearer token is right, return User instance with user role, created from header values.

    These values are passed from OpenWebUI where user signs up and authenticates.
    """
    if credentials and credentials.credentials == settings.parkupine_chat_key.get_secret_value():
        return BaseUser(
            id=openwebui_user_headers.x_openwebui_user_id,
            name=openwebui_user_headers.x_openwebui_user_name,
            email=openwebui_user_headers.x_openwebui_user_email,
            role="user",
        )
    raise HTTPException(status_code=401)


UserDep = Annotated[BaseUser, Depends(user_required)]


def admin_required(
    settings: AppSettingsDep,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> BaseUser:
    """
    If bearer token is right, returns User instance with admin role.
    """
    if credentials and credentials.credentials == settings.parkupine_admin_key.get_secret_value():
        return BaseUser(
            id="0",
            name="admin",
            email="admin@admin.com",
            role="admin",
        )
    raise HTTPException(status_code=401)


AdminDep = Annotated[BaseUser, Depends(admin_required)]
