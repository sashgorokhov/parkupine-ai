import os

import pytest
from langchain_core.stores import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver

from parkupine.agent import Agent


@pytest.fixture(scope="session")
def openai_api_key():
    try:
        return os.environ["TESTING_OPENAI_API_KEY"]
    except KeyError:
        raise EnvironmentError("Must provide TESTING_OPENAI_API_KEY environment variable to run system tests")


@pytest.fixture()
def agent(db_session, app_settings, openai_api_key):
    store = InMemoryStore()
    checkpointer = InMemorySaver()
    app_settings.parkupine_openai_api_key = openai_api_key

    agent = Agent(
        db_session=db_session,
        settings=app_settings,
        store=store,
        checkpointer=checkpointer,
    )

    return agent
