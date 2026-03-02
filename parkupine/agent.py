"""
Core agent logic
"""

import logging
import time
from typing import Iterator

from fastapi_openai_compat import ChatRequest, ChatCompletion, Choice, Message
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, AIMessage
from langchain_openai import ChatOpenAI

from parkupine.auth import BaseUser
from parkupine.settings import AppSettings

logger = logging.getLogger("uvicorn")


def handle_chat_request(chat_request: ChatRequest, user: BaseUser, settings: AppSettings) -> Iterator[ChatCompletion]:

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=settings.parkupine_openai_api_key,
    )

    agent = create_agent(
        model=llm,
        debug=settings.debug,
    )

    if chat_request.stream:
        for chunk, _ in agent.stream(
            input={"messages": chat_request.messages},
            stream_mode="messages",
        ):
            yield create_chat_completion(chunk, model=chat_request.model)
    else:
        result = agent.invoke(input={"messages": chat_request.messages})
        yield create_chat_completion(result["messages"][-1], model=chat_request.model)


def create_chat_completion(message: AIMessage, model: str) -> ChatCompletion:
    object = "chat.completion"

    delta: Message | None = None
    msg: Message | None = Message(
        role="assistant",
        content=message.content,
        tool_calls=message.tool_calls,
        refusal=message.additional_kwargs.get("refusal", None),
    )

    if isinstance(message, AIMessageChunk):
        object = "chat.completion.chunk"
        delta = msg
        msg = None

    return ChatCompletion(
        id=message.id,
        object=object,
        created=int(time.time()),
        model=model,
        usage=message.usage_metadata,
        system_fingerprint=message.response_metadata.get("system_fingerprint", None),
        choices=[
            Choice(
                index=0,
                delta=delta,
                message=msg,
                finish_reason=message.response_metadata.get("finish_reason", None),
            )
        ],
    )
