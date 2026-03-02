"""
Core agent logic
"""

import logging
import time
import uuid
from typing import Iterator

from fastapi_openai_compat import ChatRequest, ChatCompletion, Choice, Message
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage, ToolMessageChunk
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel
from sqlmodel import Session, select

from parkupine.auth import BaseUser
from parkupine.settings import AppSettings
from parkupine.tables import ParkingGarage, ParkingSpace

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a parking reservation assistant for the Parkupine parking management system.

## Core Responsibilities
You help users with:
- Creating, modifying, and canceling parking reservations
- Checking parking availability for specific dates, times, and locations
- Providing information about parking rates, hours, and policies
- Answering questions about existing reservations
- Explaining parking lot amenities and accessibility options

## Guidelines
1. Always use the provided tools to fetch real-time information about availability, pricing, and reservations
2. Never guess or make up information — if a tool doesn't provide the answer, say you don't have that information
3. Confirm important details (date, time, location, vehicle info) before finalizing any reservation
4. Provide clear pricing breakdowns when discussing costs
5. Be concise but thorough in your responses

## Boundaries
You MUST politely decline to answer questions unrelated to parking reservations. This includes but is not limited to:
- General knowledge questions
- Navigation or driving directions (beyond parking location addresses)
- Topics about other services, entertainment, or recommendations
- Personal advice or opinions

When declining, briefly explain that you're specialized in parking reservations and offer to help with
parking-related queries instead.

## Tone
Be professional, helpful, and efficient. Users are often in a hurry when managing parking, so respect their time while
ensuring accuracy.
"""

GARAGE_DETAILS_TEMPLATE = """\
Details:
{0.description}
Address: {0.address}
Working hours: {0.working_hours}
"""


class Context(BaseModel):
    user: BaseUser
    plate_number: str | None = None


class Agent:
    """
    Agent class manages graph execution and stores database connections and such.
    It also defines tools as methods that have access to provided connections.
    """

    def __init__(self, db_session: Session, checkpointer: BaseCheckpointSaver, store: BaseStore, settings: AppSettings):
        self._db_session = db_session
        self._settings = settings
        self._checkpointer = checkpointer
        self._store = store

        self._graph = self._compile_graph()

    def _compile_graph(self) -> CompiledStateGraph:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=self._settings.parkupine_openai_api_key,
        )

        # Have to do this because langchain cannot use tools created from methods.

        def list_garages_by_name() -> list[str]:
            """
            Return names of available parking garages
            """
            return self.list_garages_by_name()

        def get_garage_details_by_name(garage_name: str) -> ParkingGarage | None:
            """
            Return garage details (working hours, address, description) by name
            """
            return self.get_garage_details_by_name(garage_name)

        def list_parking_spaces_by_garage(garage_name: str) -> list[ParkingSpace]:
            """
            Return parking spaces present in garage
            """
            return self.get_parking_spaces_by_garage(garage_name)

        agent = create_agent(
            model=llm,
            debug=self._settings.debug,
            checkpointer=self._checkpointer,
            store=self._store,
            system_prompt=SYSTEM_PROMPT,
            context_schema=Context,
            tools=[list_garages_by_name, get_garage_details_by_name, list_parking_spaces_by_garage],
        )

        return agent

    def handle_chat_request(self, chat_request: ChatRequest, user: BaseUser, chat_id: str) -> Iterator[ChatCompletion]:
        context = Context(user=user)

        invoke_params = dict(
            input={"messages": chat_request.messages},
            config={"configurable": {"thread_id": chat_id}},
            context=context,
            durability="sync",
        )

        if chat_request.stream:
            for chunk, _ in self._graph.stream(**invoke_params, stream_mode="messages"):
                completion = create_chat_completion(chunk, model=chat_request.model)
                logger.debug(f"\t{type(chunk).__name__}({str(chunk)}) -> {completion.model_dump_json()}")
                yield completion
        else:
            result = self._graph.invoke(**invoke_params)
            yield create_chat_completion(result["messages"][-1], model=chat_request.model)

    def list_garages_by_name(self) -> list[str]:
        statement = select(ParkingGarage)
        garages = self._db_session.exec(statement).all()
        return [garage.name for garage in garages]

    def get_garage_details_by_name(self, garage_name: str) -> ParkingGarage | None:
        statement = select(ParkingGarage).where(ParkingGarage.name == garage_name)
        try:
            garage = self._db_session.exec(statement).one()
        except Exception:  # noqa: B001
            return None

        return garage

    def get_parking_spaces_by_garage(self, garage_name: str) -> list[ParkingSpace]:
        garage: ParkingGarage = self._db_session.exec(
            select(ParkingGarage).where(ParkingGarage.name == garage_name)
        ).one()
        spaces = self._db_session.exec(select(ParkingSpace).where(ParkingSpace.garage_id == garage.id)).all()
        return list(spaces)


def create_chat_completion(message: AIMessage | ToolMessage, model: str) -> ChatCompletion:
    """
    Convert AIMessage to ChatCompletion
    """
    object = "chat.completion"

    delta: Message | None = None
    msg: Message | None = Message(
        role="tool" if isinstance(message, ToolMessage) else "assistant",
        content=message.content,
        tool_calls=getattr(message, "tool_calls", None),
        refusal=message.additional_kwargs.get("refusal", None),
    )

    finish_reason = message.response_metadata.get("finish_reason", None)

    if isinstance(message, (AIMessageChunk, ToolMessageChunk)):
        object = "chat.completion.chunk"
        delta = msg
        msg = None

        if getattr(message, "chunk_position", None) == "last":
            finish_reason = "stop"
        else:
            finish_reason = None

    return ChatCompletion(
        id=message.id,
        object=object,
        created=int(time.time()),
        model=model,
        usage=getattr(message, "usage_metadata", None),
        system_fingerprint=message.response_metadata.get("system_fingerprint", None),
        choices=[
            Choice(
                index=0,
                delta=delta,
                message=msg,
                finish_reason=finish_reason,
            )
        ],
    )


def manual_chat_completion(message: str, model: str, stream: bool = False) -> ChatCompletion:
    object = "chat.completion"

    delta: Message | None = None
    msg: Message | None = Message(
        role="system",
        content=message,
    )

    if stream:
        object = "chat.completion.chunk"
        delta = msg
        msg = None

    return ChatCompletion(
        id=str(uuid.uuid4()),
        object=object,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                delta=delta,
                message=msg,
                finish_reason="stop",
            )
        ],
    )
