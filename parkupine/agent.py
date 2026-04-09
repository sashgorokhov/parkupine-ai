"""
Agent logic - prompts, graph definition, RAG and tool functions.

Defines Agent class, serving as a main collection of AI nonsense.
"""

import asyncio
import functools
import inspect
import logging
import time
import uuid
from typing import Iterator, Callable, Any, ParamSpec, TypeVar, Literal, Optional, cast

import fastmcp
from fastapi_openai_compat import ChatRequest, ChatCompletion, Choice, Message
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage, ToolMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from sqlalchemy.orm import sessionmaker
from sqlmodel import select, Session

from parkupine.auth import BaseUser
from parkupine.settings import AppSettings
from parkupine.tables import ParkingGarage, ParkingSpace, ParkingReservation

logger = logging.getLogger(__name__)

USER_SYSTEM_PROMPT = """\
You are Parkupine, a parking reservation assistant.

## Core Responsibilities
You help users with:
- Creating parking reservations. Before creating reservation, collect user's plate number.
- Providing details on available parking garages and their parking spaces.
- Providing information about parking rates, hours, and other details.
- Providing information about parking policy, FAQ and lot rules.

# Context
- Reservations can be made for any time period up to 7 days.
- All fees and taxes already included in price.
- Parkupine is not responsible for any damages or losses incurred during parking.
- All garages are secured with AI surveillance cameras and an army of guard porcupines.


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
Be professional and efficient, do not use bullet points or lists and act like a human.
Users are often in a hurry when managing parking, so respect their time.
"""

ADMIN_SYSTEM_PROMPT = """\
You are Parkupine, a parking reservation assistant.

## Core Responsibilities
You help users with:
- Checking currently pending parking reservations
- Processing requests of approval or rejection of parking reservations

## Boundaries
You MUST politely decline to answer questions unrelated to parking reservations.
"""

P = ParamSpec("P")
T = TypeVar("T")


class State(MessagesState):  # type: ignore[misc]
    model: str


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


def get_model_name(state: State, config: RunnableConfig) -> Literal["admin", "user"]:
    """
    Depending on user role from State context, route graph into admin or user node
    """
    user: BaseUser | None = config["configurable"].get("user", None)
    if user and user.is_admin and state["model"] == "parkupine_admin_v1":
        return "admin"
    return "user"


class Agent:
    """
    Agent class manages graph execution and stores database connections and such.
    It also defines tools as methods that have access to provided connections.

    The main entrypoint - handle_chat_request, receives a chat request and spits out LLM response tokens
    """

    def __init__(
        self,
        db_session: sessionmaker[Session],
        checkpointer: BaseCheckpointSaver,
        store: BaseStore,
        settings: AppSettings,
        vector_store: VectorStore,
        model: BaseChatModel | None = None,
        mcp_transport: fastmcp.FastMCP | str | None = None,
    ):
        self._db_session = db_session
        self._settings = settings
        self._checkpointer = checkpointer
        self._store = store
        self._vector_store = vector_store

        self._model = model or ChatOpenAI(
            model=self._settings.parkupine_openai_model,
            api_key=self._settings.parkupine_openai_api_key,
            # temperature=self._settings.parkupine_openai_temperature,
        )

        self._mcp_client = fastmcp.Client(mcp_transport or self._settings.mcp_url)

        self._graph = self._compile_graph()

    def _compile_graph(self) -> CompiledStateGraph:
        # Ideally we should use separate user and admin graphs, but for the sake of exercise
        # lets do some routing
        builder = StateGraph(state_schema=State)

        user_tools = [
            self.list_garages_by_name,
            self.get_garage_details_by_name,
            self.get_parking_spaces_by_garage,
            self.make_reservation,
            self.list_reservations,
            self.check_reservation,
            self.retrieve_context,
        ]

        admin_tools = [
            self.list_pending_reservations,
            self.reject_reservation,
            self.create_reservation_file,
        ]

        user_model = self._model.bind_tools(user_tools)
        admin_model = self._model.bind_tools(admin_tools)

        def call_user_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": [user_model.invoke([USER_SYSTEM_PROMPT] + state["messages"])]}

        def call_admin_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": [admin_model.invoke([ADMIN_SYSTEM_PROMPT] + state["messages"])]}

        builder.add_node("user", call_user_model)
        builder.add_node("admin", call_admin_model)
        builder.add_node("user_tools", ToolNode(user_tools))
        builder.add_node("admin_tools", ToolNode(admin_tools))

        builder.add_conditional_edges(START, get_model_name)
        builder.add_conditional_edges("user", tools_condition, {"tools": "user_tools", "__end__": END})
        builder.add_conditional_edges("admin", tools_condition, {"tools": "admin_tools", "__end__": END})
        builder.add_edge("user_tools", "user")
        builder.add_edge("admin_tools", "admin")

        graph = builder.compile(
            checkpointer=self._checkpointer,
            store=self._store,
            debug=self._settings.debug,
            name="Parkupine",
        )

        return graph

    def handle_chat_request(self, chat_request: ChatRequest, user: BaseUser, chat_id: str) -> Iterator[ChatCompletion]:
        """
        Run graph on provided chat request and return iterator over ChatCompletion objects
        """
        invoke_params = dict(
            input={"messages": chat_request.messages, "model": chat_request.model},
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
                # logger.debug(f"\t{type(chunk).__name__}({str(chunk)}) -> {completion.model_dump_json()}")
                yield completion
        else:
            result = self._graph.invoke(**invoke_params)
            logger.debug(f"{result}")
            yield create_chat_completion(result["messages"][-1], model=chat_request.model)

    @tool_method()
    def list_garages_by_name(self) -> list[str]:
        """
        Return names of available parking garages
        """
        with self._db_session() as session:
            statement = select(ParkingGarage)
            garages = session.exec(statement).all()
            return [garage.name for garage in garages]

    @tool_method()
    def get_garage_details_by_name(self, garage_name: str) -> ParkingGarage | None:
        """
        Return garage details (working hours, address, description) by name
        """
        with self._db_session() as session:
            statement = select(ParkingGarage).where(ParkingGarage.name == garage_name)
            try:
                garage: ParkingGarage = session.exec(statement).one()
            except Exception:  # noqa: B001
                return None

        return garage

    @tool_method()
    def get_parking_spaces_by_garage(self, garage_name: str) -> list[ParkingSpace]:
        """
        Return parking spaces present in garage
        """
        with self._db_session() as session:
            garage: ParkingGarage = session.exec(select(ParkingGarage).where(ParkingGarage.name == garage_name)).one()
            spaces = session.exec(select(ParkingSpace).where(ParkingSpace.garage_id == garage.id)).all()
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
        with self._db_session() as session:
            parking_space = session.exec(select(ParkingSpace).where(ParkingSpace.name == parking_space_name)).one()

            reservation = ParkingReservation(
                user_id=config["configurable"]["user"].id,
                user_name=user_name,
                user_surname=user_surname,
                plate_number=plate_number,
                period=reservation_period,
                space_id=parking_space.id,
            )
            session.add(reservation)
            session.commit()
            session.refresh(reservation)

        return f"Created reservation with id={reservation.id}. Status: {reservation.status}"

    @tool_method()
    def list_reservations(self, config: RunnableConfig) -> list[ParkingReservation]:
        """
        Return all current reservations
        """
        with self._db_session() as session:
            return list(
                session.exec(
                    select(ParkingReservation).where(ParkingReservation.user_id == config["configurable"]["user"].id)
                ).all()
            )

    @tool_method()
    def check_reservation(self, config: RunnableConfig, reservation_id: int) -> str:
        """
        Check reservation status
        """
        with self._db_session() as session:
            reservation: Optional[ParkingReservation] = session.exec(
                select(ParkingReservation)
                .where(ParkingReservation.id == reservation_id)
                .where(ParkingReservation.user_id == config["configurable"]["user"].id)
            ).one_or_none()
            if not reservation:
                return "Reservation not found"
            return reservation.status

    @tool_method()
    def list_pending_reservations(self) -> list[ParkingReservation] | str:
        """
        Return all parking reservations that are pending review or require admin/manager attention
        """
        with self._db_session() as session:
            reservations = list(
                session.exec(select(ParkingReservation).where(ParkingReservation.status == "on_review")).all()
            )

        if not reservations:
            return "No pending reservations"
        return reservations

    # Ideally this logic must be park of Service layer
    def set_reservation_status(self, reservation_id: int, status: str) -> ParkingReservation:
        """
        Approve a pending parking reservation by setting its status to approved
        """
        with self._db_session() as session:
            reservation: Optional[ParkingReservation] = session.exec(
                select(ParkingReservation).where(ParkingReservation.id == reservation_id).with_for_update()
            ).one_or_none()

            if not reservation:
                raise ValueError("Reservation not found")

            reservation.status = status

            session.add(reservation)
            session.commit()
            session.refresh(reservation)

            return reservation

    @tool_method()
    def create_reservation_file(
        self, reservation_id: int, user_name: str, user_surname: str, plate_number: str, period: str
    ) -> str:
        """
        Approve a pending parking reservation
        """

        self.set_reservation_status(reservation_id, "approved")

        # This is needed because both fastmcp and langchain's MCP code does not work in synchronous context.
        # This is a huge hurdle that requires full system redesign to handle properly
        async def async_context() -> str:
            async with self._mcp_client:
                result = await self._mcp_client.call_tool(
                    "create_reservation_file",
                    {
                        "user_name": user_name,
                        "user_surname": user_surname,
                        "plate_number": plate_number,
                        "period": period,
                    },
                )
                return cast(str, result.data["status"])

        return asyncio.run(async_context())

    @tool_method()
    def reject_reservation(self, reservation_id: int) -> str:
        """
        Reject a pending parking reservation by its reservation_id
        """
        try:
            self.set_reservation_status(reservation_id, "rejected")
            return f"Reservation {reservation_id} has been rejected"
        except ValueError:
            return "Reservation not found"

    @tool_method(response_format="content_and_artifact")
    def retrieve_context(self, query: str) -> tuple[str, list[Document]]:
        """Retrieve information to help answer users question about FAQ, lot rules and policy"""
        retrieved_docs = self._vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(doc.page_content for doc in retrieved_docs)
        logger.debug(f'RAG query: "{query}" response: {serialized.replace("\n", " ")}')
        return serialized, retrieved_docs


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
