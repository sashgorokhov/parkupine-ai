import contextlib
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from fastapi.routing import _DefaultLifespan
from fastapi_openai_compat import ChatRequest
from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from parkupine.auth import BaseUser
from parkupine.context import AppContext
from parkupine.server import app
from fastapi.testclient import TestClient

from parkupine.settings import AppSettings


@pytest.fixture()
def app_settings():
    return AppSettings(redis_url="redis://localhost", parkupine_chat_key="test", parkupine_openai_api_key="test")


@pytest.fixture()
def redis():
    m = AsyncMock()

    @contextlib.asynccontextmanager
    async def pubsub():
        yield m.pubsub_mock

    m.pubsub = pubsub

    return m


@pytest.fixture()
def app_context(app_settings, redis):
    return AppContext(app=app, settings=app_settings, redis=redis)


@pytest.fixture()
def client(app_context):
    with mock.patch.object(app.router, "lifespan_context", _DefaultLifespan(app.router)):
        with TestClient(app) as client:
            yield client


@pytest.fixture()
def set_auth_headers(client, app_settings):
    client.headers["Authorization"] = f"Bearer {app_settings.parkupine_chat_key.get_secret_value()}"
    client.headers["x-openwebui-chat-id"] = "test-chat-id"
    client.headers["x-openwebui-user-name"] = "test-user-name"
    client.headers["x-openwebui-user-email"] = "test-user-email@example.com"
    client.headers["x-openwebui-user-id"] = "test-user-id"


@pytest.fixture()
def user():
    return BaseUser(id="test_id", name="test_name", email="test_email@example.com", role="user")


@pytest.fixture()
def chat_request():
    return ChatRequest(model="test", messages=[{"role": "user", "text": "text"}], stream=False)
