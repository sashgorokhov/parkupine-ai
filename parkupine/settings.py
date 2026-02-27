from pydantic_settings import BaseSettings, SettingsConfigDict

APP_TITLE = "Parkupine"
APP_DESCRIPTION = "Parkupine - your parking reservation system"


class AppSettings(BaseSettings):
    debug: bool = False

    parkupine_chat_key: str
    parkupine_openai_api_key: str
    parkupine_openai_model: str = "gpt-4"

    model_id: str = "parkupine_v1"
    model_name: str = "Parkupine"
    model_owner: str = "Parkupine Inc."

    parkupine_admin_key: str = "admin"

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="",
    )
