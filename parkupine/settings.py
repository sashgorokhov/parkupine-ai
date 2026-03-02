import logging.config
import os
import sys

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

    database_host: str = "localhost"
    database_port: int = 5432
    database_user: str = "parkupine"
    database_password: SecretStr = SecretStr("password")
    database_name: str = "parkupine"

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="",
    )

    @property
    def database_url(self) -> SecretStr:
        return SecretStr(
            f"postgresql+psycopg2://{self.database_user}:{self.database_password.get_secret_value()}@{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def database_url_pg3(self) -> SecretStr:
        """For psycopg3 used by langchain"""
        return SecretStr(
            f"postgresql://{self.database_user}:{self.database_password.get_secret_value()}@{self.database_host}:{self.database_port}/{self.database_name}"
        )


# Sets up just stdout handler since we are docker-native anyway
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": sys.stderr,
        },
    },
    "loggers": {
        "parkupine": {
            "level": "DEBUG",
            "handlers": ["stdout"],
            "propagate": False,
        },
        # When modules executed directly (debugging, script usage) logger name will be set to this
        "__main__": {
            "level": "DEBUG",
            "handlers": ["stdout"],
            "propagate": False,
        },
    },
}


def setup_logging() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)
