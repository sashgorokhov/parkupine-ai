import contextlib
from typing import Any, Sequence, Callable
from unittest import mock
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.routing import _DefaultLifespan
from fastapi.testclient import TestClient
from fastapi_openai_compat import ChatRequest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import GenericFakeChatModel, BaseChatModel, LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.base import coerce_to_runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field
from sqlmodel import Session, create_engine

from parkupine import tables
from parkupine.agent import Agent
from parkupine.auth import BaseUser
from parkupine.context import AppContext
from parkupine.server import app
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
    return ChatRequest(model="test", messages=[{"role": "user", "content": "text"}], stream=False)


@pytest.fixture()
def engine():
    engine = create_engine("sqlite:///:memory:")
    tables.populate_metadata(engine)
    return engine


@pytest.fixture()
def db_session(engine):
    with Session(engine) as session:
        yield session


class MockChatModel(BaseChatModel):
    """
    Special model for unittests.

    Use .mock._generate.side_effect to specify returned messages
    """

    mock: Mock | None = Field(default=None)

    def __init__(self, m: Mock = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock = m or Mock()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.mock._generate(messages)
        if isinstance(message, Mock):
            raise NotImplementedError("Set mock._generate to return a string or AIMessage")
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__

    def bind_tools(
        self,
        tools: Sequence[builtins.dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


@pytest.fixture()
def model():
    return MockChatModel()


@pytest.fixture()
def agent(db_session, app_settings, model):
    store = InMemoryStore()
    checkpointer = InMemorySaver()

    return Agent(db_session=db_session, store=store, checkpointer=checkpointer, settings=app_settings, model=model)
