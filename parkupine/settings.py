import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_TITLE = "Parkupine"
APP_DESCRIPTION = "Parkupine - your parking reservation system"
DEBUG = "DEBUG" in os.environ


class AppSettings(BaseSettings):
    debug: bool = DEBUG

    redis_url: str

    parkupine_chat_key: SecretStr
    parkupine_openai_api_key: SecretStr
    parkupine_openai_model: str = "gpt-4"

    model_id: str = "parkupine_v1"
    model_name: str = "Parkupine"
    model_owner: str = "Parkupine Inc."

    parkupine_admin_key: SecretStr = SecretStr("admin")

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="",
    )
