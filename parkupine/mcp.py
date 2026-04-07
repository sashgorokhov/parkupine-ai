"""
MCP server tools for parking reservations
"""

import datetime

from fastmcp import FastMCP

mcp = FastMCP("Parking Reservations")

RESERVATION_FILE = "/mnt/reservations.csv"


def write_reservation_file(user_name: str, user_surname: str, plate_number: str, period: str) -> None:
    """Write reservation data to file"""
    with open(RESERVATION_FILE, "a") as file:
        file.write(f"{user_name} {user_surname}|{plate_number}|{period}|{datetime.datetime.utcnow().isoformat()}\n")


@mcp.tool  # type: ignore[misc]
def create_reservation_file(user_name: str, user_surname: str, plate_number: str, period: str) -> dict[str, str]:
    """Create reservation file"""
    write_reservation_file(user_name, user_surname, plate_number, period)
    return {"status": "created"}


app = mcp.http_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
