"""
Defines Worker class that runs in a separate process. It polls redis queue
for ChatWorkItem to work on, and once received, forwards that information to Agent.

Tokens returned by agent are streamed back into redis channel.
"""

import asyncio
import logging
import uuid
from typing import Any, Optional, AsyncIterator

from fastapi_openai_compat import ChatRequest, ChatCompletion
from fastapi_openai_compat.streaming import _completion_to_sse
from langgraph.checkpoint.postgres import ShallowPostgresSaver
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel
from redis import Redis
from sqlmodel import Session
from sqlalchemy.orm import sessionmaker
from parkupine.agent import Agent, manual_chat_completion
from parkupine.auth import BaseUser
from parkupine.settings import AppSettings, setup_logging
from parkupine.tables import get_engine

logger = logging.getLogger(__name__)


class ChatWorkItem(BaseModel):
    message_id: str
    chat_request: ChatRequest
    user: BaseUser
    chat_id: str


GENERATION_COMPLETE = "_GENERATION_COMPLETE_"


async def submit_chat_request(
    redis: Redis, chat_request: ChatRequest, user: BaseUser, chat_id: str
) -> AsyncIterator[ChatCompletion | str]:
    """
    Send ChatRequest and related metadata into Worker through queue, and subscribe to its response through redis
    channel.
    """
    message_id = f"{user.id}-{chat_id}-{uuid.uuid4()}"

    work_item = ChatWorkItem(
        message_id=message_id,
        chat_request=chat_request,
        user=user,
        chat_id=chat_id,
    )

    logger.info(f"Submitting chat work item {work_item.message_id=}")
    # Worker will watch for this queue for new work items
    await redis.lpush("chat_requests", work_item.model_dump_json())

    async with redis.pubsub() as ps:
        await ps.subscribe(work_item.message_id)

        try:
            async with asyncio.timeout(30.0):
                while True:
                    # {"type": ..., "pattern": ..., "channel": ..., "data": ...}
                    message: dict[str, Any] | None = await ps.get_message(ignore_subscribe_messages=True, timeout=10.0)

                    if not message:
                        continue

                    if message["data"] == GENERATION_COMPLETE.encode():
                        return

                    chat_completion = ChatCompletion.model_validate_json(message["data"])

                    if chat_request.stream:
                        yield _completion_to_sse(chat_completion)
                    else:
                        yield chat_completion
        except TimeoutError:
            logger.error(f"Timed out while waiting for model response {message_id=}")
            completion = manual_chat_completion(
                message=f"Timed out while waiting for model response. Trace id: {message_id}",
                model=chat_request.model,
                stream=True,
            )
            completion.choices[0].finish_reason = "stop"
            if chat_request.stream:
                yield _completion_to_sse(completion)
            else:
                yield completion


class Worker:
    """
    Worker that processes chat requests from Redis queue and publishes results to channels.

    Usage:
        with Worker(agent, redis) as worker:
            worker.start()
    """

    def __init__(self, agent: Agent, redis: Redis) -> None:
        self.agent = agent
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

            chat_completions = self.agent.handle_chat_request(
                chat_request=chat_work_item.chat_request, chat_id=chat_work_item.chat_id, user=chat_work_item.user
            )

            for completion in chat_completions:
                serialized = completion.model_dump_json()
                self.redis.publish(chat_work_item.message_id, serialized)

        except Exception:
            logger.exception(f"Error processing message {chat_work_item.message_id=}")
            logger.debug(chat_work_item.model_dump_json())
            completion = manual_chat_completion(
                message=f"Model execution failed. Trace id: {chat_work_item.message_id}",
                model=chat_work_item.chat_request.model,
                stream=chat_work_item.chat_request.stream,
            )
            completion.choices[0].finish_reason = "stop"
            self.redis.publish(chat_work_item.message_id, completion.model_dump_json())
        finally:
            self.redis.publish(chat_work_item.message_id, GENERATION_COMPLETE)
            logger.debug(f"Finished {chat_work_item.message_id=}")


def entrypoint() -> None:
    """
    Worker entrypoint method, sets up environment and runs worker logic
    """
    settings = AppSettings()
    redis = Redis.from_url(settings.redis_url)
    engine = get_engine(settings)
    SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False)

    logger.info(f"Initializing: {settings}")

    postgres_url = settings.database_url_pg3.get_secret_value()

    with (
        ShallowPostgresSaver.from_conn_string(postgres_url) as checkpointer,
        PostgresStore.from_conn_string(postgres_url) as store,
    ):

        agent = Agent(db_session=SessionLocal, checkpointer=checkpointer, store=store, settings=settings)
        worker = Worker(agent=agent, redis=redis)

        with worker:
            worker.start()


if __name__ == "__main__":
    setup_logging()
    entrypoint()
