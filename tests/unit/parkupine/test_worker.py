from unittest.mock import Mock

import pytest
from redis import Redis

from parkupine.agent import manual_chat_completion, Agent
from parkupine.worker import submit_chat_request, GENERATION_COMPLETE, ChatWorkItem, Worker


@pytest.fixture()
def chat_work_item(user, chat_request):
    return ChatWorkItem(
        message_id="test-message-id",
        chat_id="test-chat-id",
        user=user,
        chat_request=chat_request,
    )


@pytest.fixture()
def agent():
    return Mock(spec=Agent)


@pytest.fixture()
def sync_redis():
    return Mock(spec=Redis)


@pytest.fixture()
def worker(agent, sync_redis):
    return Worker(agent=agent, redis=sync_redis)


@pytest.mark.asyncio
async def test_submit_chat_request(redis, user, chat_request):
    expected_chat_completion = manual_chat_completion(
        message="test",
        model="test",
        stream=False,
    )

    redis.pubsub_mock.get_message.side_effect = [
        None,
        {"data": expected_chat_completion.model_dump_json().encode()},
        {"data": GENERATION_COMPLETE.encode()},
    ]

    iterator = submit_chat_request(redis=redis, chat_request=chat_request, user=user, chat_id="test")

    chat_completion = await anext(iterator)

    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert chat_completion == expected_chat_completion


def test_worker_start(chat_work_item, worker, agent, sync_redis):
    expected_completion = manual_chat_completion(
        message="test",
        model="test",
        stream=False,
    )
    agent.handle_chat_request.side_effect = [[expected_completion]]

    sync_redis.brpop.side_effect = [None, [], ["test", chat_work_item.model_dump_json()], KeyboardInterrupt()]

    with worker:
        worker.start()

    assert sync_redis.publish.call_count == 2

    sync_redis.publish.assert_any_call(chat_work_item.message_id, expected_completion.model_dump_json())
    sync_redis.publish.assert_any_call(chat_work_item.message_id, GENERATION_COMPLETE)


def test_worker_handle_chat_work_item_error(chat_work_item, worker, agent, sync_redis):
    agent.handle_chat_request.side_effect = [ValueError("test")]

    worker.handle_chat_work_item(chat_work_item)

    assert sync_redis.publish.call_count == 2

    sync_redis.publish.assert_any_call(chat_work_item.message_id, GENERATION_COMPLETE)
