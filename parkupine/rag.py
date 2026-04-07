"""
RAG indexing pipeline. populate_vector_store is called during initial data population of postgres in tables.py script.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import Engine

from parkupine.settings import AppSettings


# List of documents to be loaded into vector store for POC
DOCUMENTS: list[tuple[str, str]] = [
    (
        "policy",
        """
    Parking reservations are valid only for the selected date, time window, and parking zone shown in the booking confirmation. Vehicles arriving more than 15 minutes early may be asked to wait until the reserved start time, and vehicles arriving more than 30 minutes late may forfeit the reservation without refund if the space has been reallocated. Customers must display a valid reservation confirmation or license plate match must be present in the system at entry. Each reservation covers one standard passenger vehicle unless an oversized vehicle category was explicitly booked.
    Cancellations made at least 24 hours before the reservation start time are eligible for a full refund. Cancellations made within 24 hours may be subject to a service fee, and no-shows are not refundable. If a parking facility becomes unavailable due to maintenance, weather, or safety concerns, Parkupine may cancel the reservation and issue a full refund or rebook the customer at an equivalent nearby location when available. Users are responsible for ensuring their vehicle fits the posted height, length, and weight limits for the selected facility.
    Parking must comply with all posted site rules, including speed limits, accessible-space restrictions, EV charging usage rules, and time limits for loading zones. Vehicles left beyond the reserved period may incur additional hourly fees or towing at the owner’s expense, depending on facility policy. Parkupine reserves the right to suspend reservation privileges for repeated violations, fraud, or misuse of the booking system.
    """,  # noqa
    ),
    (
        "faq",
        """
    Q: How do I find my reservation when I arrive?
    A: You can access your reservation in the Parkupine app or by using the confirmation email sent after booking. Most facilities also recognize the license plate attached to your reservation, so you may be able to enter without scanning a QR code. If the gate does not open automatically, use the intercom or help button and provide your reservation ID.
    Q: Can I change my reservation after booking?
    A: In most cases, yes. You can modify your start time, end time, or parking location if the new slot is available. Changes made close to the reservation time may include a price difference or a small modification fee. If you need to shorten your stay, Parkupine will show whether a partial refund is available before you confirm the change.
    Q: What happens if I arrive early, late, or need help on-site?
    A: Reservations are held for a limited grace period, but arriving significantly early or late may affect access to your space. If your vehicle does not fit the facility’s posted size limits, staff may direct you to an alternate area or deny entry for safety reasons. For assistance, contact on-site support through the app or facility help desk, and include your reservation number so the team can help quickly.
    """,  # noqa
    ),
    (
        "lot rules",
        """
    All vehicles must park only within the assigned space or zone shown on the reservation confirmation. Parking across lines, in fire lanes, at entrances, or in reserved-access areas is prohibited. Drivers must follow all posted signs, speed limits, directional arrows, and staff instructions while entering, exiting, and moving through the facility. Vehicles left in unauthorized areas may be relocated or cited at the owner’s expense.
    The parking facility is intended for standard passenger vehicles unless an oversized, electric, accessible, or specialty vehicle category was explicitly booked. Vehicles must meet the posted height, length, width, and weight restrictions for the lot. Trailers, commercial vehicles, and vehicles carrying hazardous materials are not allowed unless the facility description clearly says otherwise. If a vehicle cannot safely fit in the assigned space, Parkupine may cancel or reassign the reservation without guaranteeing the original spot.
    Guests are responsible for securing their own vehicles and personal belongings. Parkupine and the parking operator are not liable for theft, loss, or damage caused by unattended valuables left inside the vehicle. Windows should be closed, doors locked, and the engine turned off before leaving the car. Any alarms, leaks, or mechanical issues that affect neighboring vehicles or facility operations must be reported immediately.
    Idle parking, double parking, and blocking another vehicle’s exit are strictly prohibited. Drivers may not use the lot for vehicle storage, repairs, washing, fueling, or loading that exceeds the posted time limits. Electric vehicle charging stations may be used only while actively charging and only for the duration allowed by the reservation or facility rules. Repeated violations may result in fees, towing, or suspension of reservation privileges.
    Users must leave the lot before the reservation ends unless an extension is confirmed in the system. Overstaying the booked time may result in additional hourly charges or enforcement action depending on the location’s policy. If a reservation is canceled, modified, or reassigned, the updated confirmation always takes priority over any earlier instructions. Parkupine reserves the right to update these rules when needed for safety, operations, or compliance.
    """,  # noqa
    ),
]


def get_vector_store(engine: Engine, settings: AppSettings, embeddings: Embeddings | None = None) -> VectorStore:
    embeddings = embeddings or OpenAIEmbeddings(
        model=settings.embeddings_model,
        api_key=settings.parkupine_openai_api_key,
    )

    return PGVector(embeddings=embeddings, collection_name="parkupine", connection=engine)


def populate_vector_store(settings: AppSettings, engine: Engine) -> None:
    vector_store = get_vector_store(engine=engine, settings=settings)

    ids, texts = zip(*DOCUMENTS)

    vector_store.delete()

    vector_store.add_texts(
        texts=texts,
        ids=ids,
    )
