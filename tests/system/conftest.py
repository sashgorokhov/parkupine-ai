import os

import pytest
from langchain_core.stores import InMemoryStore

# This injects vcr_config fixture into our conftest
from langchain_tests.conftest import vcr_config  # noqa
from langgraph.checkpoint.memory import InMemorySaver
from testcontainers.postgres import PostgresContainer
from sqlmodel import Session, create_engine
from sqlalchemy.orm import sessionmaker
from parkupine.agent import Agent
from parkupine.rag import get_vector_store


@pytest.fixture()
def postgres_container():
    with PostgresContainer(
        image="pgvector/pgvector:pg16",
    ) as postgres_container:
        yield postgres_container


@pytest.fixture()
def engine(postgres_container):
    return create_engine(postgres_container.get_connection_url())


@pytest.fixture()
def openai_api_key(monkeypatch, app_settings):
    try:
        key = os.environ["TESTING_OPENAI_API_KEY"]
    except KeyError:
        raise EnvironmentError("Must provide TESTING_OPENAI_API_KEY environment variable to run system tests")

    monkeypatch.setenv("OPENAI_API_KEY", key)
    app_settings.parkupine_openai_api_key = key
    yield key


@pytest.fixture()
def db_session(engine):
    return sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False)


@pytest.fixture()
def agent(db_session, engine, app_settings, openai_api_key):
    store = InMemoryStore()
    checkpointer = InMemorySaver()
    app_settings.parkupine_openai_api_key = openai_api_key

    vector_store = get_vector_store(engine, app_settings)

    agent = Agent(
        db_session=db_session,
        settings=app_settings,
        store=store,
        checkpointer=checkpointer,
        vector_store=vector_store,
    )

    return agent
