"""
ORM models for parking reservation entities like ParkingGarage and ParkingSpace.

ParkingGarage describes parking garage/lot, and acts as a "container" for any number of parking spaces.

This information used by RAG pipeline.
"""

import logging

from langgraph.checkpoint.postgres import ShallowPostgresSaver
from langgraph.store.postgres import PostgresStore
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel, Field, create_engine, Session

from parkupine.settings import AppSettings, setup_logging


logger = logging.getLogger(__name__)


class ParkingGarage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    description: str = Field()
    address: str = Field()
    working_hours: str = Field()


class ParkingSpace(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    garage_id: int | None = Field(default=None, foreign_key="parkinggarage.id")
    name: str = Field()
    price: int = Field()


def get_engine(settings: AppSettings) -> Engine:
    return create_engine(settings.database_url.get_secret_value())


def populate_metadata(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


def populate_data(engine: Engine) -> None:
    """
    Populate ParkingGarage and ParkingSpace tables with some data. This call must be idempotent!
    """
    # Note the hardcoded PKs
    garages = [
        ParkingGarage(
            id=1,
            name="Green Garage",
            description="Spacious garage near town business center.",
            address="123 Main Street",
            working_hours="24/7",
        ),
        ParkingGarage(
            id=2,
            name="Blue Garage",
            description="Small garage near Dillards.",
            address="56 Hairy Man road",
            working_hours="9AM - 6PM Mon-Fri",
        ),
    ]
    spaces = [
        ParkingSpace(id=1, garage_id=1, name="1G", price=10),
        ParkingSpace(id=2, garage_id=1, name="2G", price=10),
        ParkingSpace(id=3, garage_id=1, name="3G", price=10),
        ParkingSpace(id=4, garage_id=2, name="10B", price=20),
        ParkingSpace(id=5, garage_id=2, name="20B", price=20),
    ]

    with Session(engine) as s:
        for garage in garages:
            logger.debug(f"Populating: {garage}")
            s.add(garage)

        for space in spaces:
            logger.debug(f"Populating: {space}")
            s.add(space)

        try:
            s.commit()
        except IntegrityError:
            pass


def setup_langgraph_resources(settings: AppSettings) -> None:
    logger.debug("Setting up postgres checkpointer")
    with ShallowPostgresSaver.from_conn_string(settings.database_url_pg3.get_secret_value()) as checkpointer:
        checkpointer.setup()

    logger.debug("Setting up postgres store")
    with PostgresStore.from_conn_string(settings.database_url_pg3.get_secret_value()) as store:
        store.setup()


# Simple entrypoint to create baseline data in postgres
if __name__ == "__main__":
    setup_logging()
    settings = AppSettings()
    engine = get_engine(settings)

    populate_metadata(engine)
    populate_data(engine)
    setup_langgraph_resources(settings)
