import asyncio

import pytest
from testcontainers.redis import AsyncRedisContainer


@pytest.fixture()
def redis_container():
    with AsyncRedisContainer() as container:
        yield container


@pytest.fixture()
def redis(redis_container):
    client = asyncio.run(redis_container.get_async_client())
    asyncio.run(client.initialize())
    yield client


@pytest.fixture()
def sync_redis(redis_container):
    with redis_container.get_client() as client:
        yield client
