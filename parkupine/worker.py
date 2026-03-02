""" """

import asyncio
import logging
import uuid
from typing import Any, Optional, AsyncIterator

from fastapi_openai_compat import ChatRequest, ChatCompletion
from fastapi_openai_compat.streaming import _completion_to_sse
from pydantic import BaseModel
from redis import Redis

from parkupine.agent import handle_chat_request
from parkupine.auth import BaseUser
from parkupine.settings import AppSettings

logger = logging.getLogger(__name__)


class ChatWorkItem(BaseModel):
    message_id: str
    chat_request: ChatRequest
    user: BaseUser


async def submit_chat_request(
    redis: Redis, chat_request: ChatRequest, user: BaseUser
) -> AsyncIterator[ChatCompletion | str]:
    """
    Send ChatRequest and related metadata into Worker through queue, and subscribe to its response through redis
    channel.
    """
    message_id = f"{user.id}-{uuid.uuid4()}"

    work_item = ChatWorkItem(
        message_id=message_id,
        chat_request=chat_request,
        user=user,
    )

    logger.info(f"Submitting chat work item {work_item.message_id=}")
    # Worker will watch for this queue for new work items
    await redis.lpush("chat_requests", work_item.model_dump_json())

    async with redis.pubsub() as ps:
        await ps.subscribe(work_item.message_id)

        while True:
            # {"type": ..., "pattern": ..., "channel": ..., "data": ...}
            try:
                message: dict[str, Any] = await ps.get_message(ignore_subscribe_messages=True, timeout=10.0)
            except asyncio.TimeoutError:
                logger.error(f"Timed out while waiting for chat message response at {work_item.message_id=}")
                # Perhaps we should return artificial ChatCompletion with something like "Model Timed Out"
                return

            if not message:
                continue

            chat_completion = ChatCompletion.model_validate_json(message["data"])

            if chat_request.stream:
                yield _completion_to_sse(chat_completion)
            else:
                yield chat_completion

            if chat_completion.choices and chat_completion.choices[-1].finish_reason is not None:
                logger.debug(f"Generation complete for {work_item.message_id=}")
                break


class Worker:
    """
    Worker that processes chat requests from Redis queue and publishes results to channels.

    This is supposed to run as a separate process from main server.

    Usage:
        with Worker(settings, redis) as worker:
            worker.start()
    """

    def __init__(self, settings: AppSettings, redis: Redis) -> None:
        """
        Initialize worker

        Args:
            settings: Application settings containing Redis URL and other config
        """
        self.settings = settings
        self.redis = redis
        self.running = False

    def start(self) -> None:
        """Start the worker to process messages from the queue"""
        self.running = True
        while self.running:
            try:
                message_body: Optional[list[str]] = self.redis.brpop(["chat_requests"], timeout=1)
                if message_body:
                    chat_work_item = ChatWorkItem.model_validate_json(message_body[1])
                    self.handle_chat_work_item(chat_work_item=chat_work_item)
            except KeyboardInterrupt:
                logger.info("Worker interrupted by user")
                self.running = False
            except TimeoutError:
                pass
            except Exception:
                logger.exception("Error processing message")

    def stop(self) -> None:
        """Stop the worker"""
        self.running = False
        self.redis.close()
        logger.info("Worker stopped")

    def __enter__(self) -> "Worker":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        self.stop()

    def handle_chat_work_item(self, chat_work_item: ChatWorkItem) -> None:
        """
        Process ChatWorkItem received from queue - run agent and publish its result tokens into channel.
        """
        try:
            logger.debug(f"Starting {chat_work_item.message_id=}")

            chat_completions = handle_chat_request(
                chat_request=chat_work_item.chat_request, user=chat_work_item.user, settings=self.settings
            )

            for completion in chat_completions:
                serialized = completion.model_dump_json()
                self.redis.publish(chat_work_item.message_id, serialized)

            logger.debug(f"Finished {chat_work_item.message_id=}")
        except Exception:
            logger.debug(str(chat_work_item.chat_request))
            logger.exception(f"Error processing message {chat_work_item.message_id=}")


def entrypoint() -> None:
    """
    Worker entrypoint method, sets up environment and runs worker logic
    """
    settings = AppSettings()
    redis = Redis.from_url(settings.redis_url)

    logger.info(f"Initializing: {settings}")

    with Worker(settings=settings, redis=redis) as worker:
        worker.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    entrypoint()
