"""
Agent logic - prompts, graph definition, RAG and tool functions.

Defines Agent class, serving as a main collection of AI nonsense.
"""

import functools
import inspect
import logging
import time
import uuid
from typing import Iterator, Callable, Any, ParamSpec, TypeVar

from fastapi_openai_compat import ChatRequest, ChatCompletion, Choice, Message
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage, ToolMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from sqlmodel import Session, select

from parkupine.auth import BaseUser
from parkupine.settings import AppSettings
from parkupine.tables import ParkingGarage, ParkingSpace, ParkingReservation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Parkupine, a parking reservation assistant.

## Core Responsibilities
You help users with:
- Creating parking reservations. Before creating reservation, collect user's plate number.
- Providing details on available parking garages and their parking spaces.
- Providing information about parking rates, hours, and other details.

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

P = ParamSpec("P")
T = TypeVar("T")


def tool_method(**kwargs: Any) -> Callable[[Callable[P, T]], property]:
    """
    Create langchain Tool from class method.
    This is needed because langchain cannot work with functions that have `self` argument (or any other non-llm
    provided for that matter).

    Basically what it does it hides original method funder other function while copying its name, signature
    (excluding first argument - self) and docstring.

    :param kwargs: passed to @tool decorator
    """

    def decorator(func: Callable[P, T]) -> property:
        @property  # type: ignore[misc]
        def wrapper(self: Any) -> BaseTool:
            sig = inspect.signature(func)
            new_params = list(sig.parameters.values())[1:]
            new_sig = sig.replace(parameters=new_params)

            @functools.wraps(func)
            def bound(*args: Any, **kwargs: Any) -> T:
                return func(self, *args, **kwargs)

            bound.__signature__ = new_sig  # type: ignore[attr-defined]
            return tool(bound, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


class Agent:
    """
    Agent class manages graph execution and stores database connections and such.
    It also defines tools as methods that have access to provided connections.

    The main entrypoint - handle_chat_request, receives a chat request and spits out LLM response tokens
    """

    def __init__(
        self,
        db_session: Session,
        checkpointer: BaseCheckpointSaver,
        store: BaseStore,
        settings: AppSettings,
        model: BaseChatModel | None = None,
    ):
        self._db_session = db_session
        self._settings = settings
        self._checkpointer = checkpointer
        self._store = store

        model = model or ChatOpenAI(
            model=self._settings.parkupine_openai_model,
            api_key=self._settings.parkupine_openai_api_key,
            temperature=self._settings.parkupine_openai_temperature,
        )

        self._graph = self._compile_graph(model)

    def _compile_graph(self, model: BaseChatModel) -> CompiledStateGraph:
        agent = create_agent(
            model=model,
            debug=self._settings.debug,
            checkpointer=self._checkpointer,
            store=self._store,
            system_prompt=SYSTEM_PROMPT,
            tools=[
                self.list_garages_by_name,
                self.get_garage_details_by_name,
                self.get_parking_spaces_by_garage,
                self.make_reservation,
                self.list_reservations,
                self.check_reservation,
            ],
        )

        return agent

    def handle_chat_request(self, chat_request: ChatRequest, user: BaseUser, chat_id: str) -> Iterator[ChatCompletion]:
        """
        Run graph on provided chat request and return iterator over ChatCompletion objects
        """
        invoke_params = dict(
            input={"messages": chat_request.messages},
            config=RunnableConfig(
                configurable={
                    "thread_id": chat_id,
                    "user": user,
                    "chat_id": chat_id,
                }
            ),
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

    @tool_method()
    def list_garages_by_name(self) -> list[str]:
        """
        Return names of available parking garages
        """
        statement = select(ParkingGarage)
        garages = self._db_session.exec(statement).all()
        return [garage.name for garage in garages]

    @tool_method()
    def get_garage_details_by_name(self, garage_name: str) -> ParkingGarage | None:
        """
        Return garage details (working hours, address, description) by name
        """
        statement = select(ParkingGarage).where(ParkingGarage.name == garage_name)
        try:
            garage = self._db_session.exec(statement).one()
        except Exception:  # noqa: B001
            return None

        return garage

    @tool_method()
    def get_parking_spaces_by_garage(self, garage_name: str) -> list[ParkingSpace]:
        """
        Return parking spaces present in garage
        """
        garage: ParkingGarage = self._db_session.exec(
            select(ParkingGarage).where(ParkingGarage.name == garage_name)
        ).one()
        spaces = self._db_session.exec(select(ParkingSpace).where(ParkingSpace.garage_id == garage.id)).all()
        return list(spaces)

    @tool_method()
    def make_reservation(
        self,
        config: RunnableConfig,
        plate_number: str,
        user_name: str,
        user_surname: str,
        parking_space_name: str,
        reservation_period: str,
    ) -> str:
        """
        Create a reservation for parking space.
        Must ask user for:
        - parking space name for reservation
        - name and surname
        - car plate number
        - reservation period
        """
        parking_space = self._db_session.exec(select(ParkingSpace).where(ParkingSpace.name == parking_space_name)).one()

        reservation = ParkingReservation(
            user_id=config["configurable"]["user"].id,
            user_name=user_name,
            user_surname=user_surname,
            plate_number=plate_number,
            period=reservation_period,
            space_id=parking_space.id,
        )
        self._db_session.add(reservation)
        self._db_session.commit()
        self._db_session.refresh(reservation)

        return f"Created reservation with id={reservation.id}. Status: {reservation.status}"

    @tool_method()
    def list_reservations(self, config: RunnableConfig) -> list[ParkingReservation]:
        """
        Return all current reservations
        """
        return list(
            self._db_session.exec(
                select(ParkingReservation).where(ParkingReservation.user_id == config["configurable"]["user"].id)
            ).all()
        )

    @tool_method()
    def check_reservation(self, config: RunnableConfig, reservation_id: int) -> str:
        """
        Check reservation status
        """
        reservation = self._db_session.exec(
            select(ParkingReservation)
            .where(ParkingReservation.id == reservation_id)
            .where(ParkingReservation.user_id == config["configurable"]["user"].id)
        ).one_or_none()
        if not reservation:
            return "Reservation not found"
        return reservation.status


def create_chat_completion(message: AIMessage | ToolMessage, model: str) -> ChatCompletion:
    """
    Convert AIMessage to ChatCompletion
    """
    object = "chat.completion"

    delta: Message | None = None
    msg: Message | None = Message(
        role="tool" if isinstance(message, ToolMessage) else "assistant",
        content=message.content or "",
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
