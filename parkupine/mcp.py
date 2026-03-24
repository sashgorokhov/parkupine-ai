"""
MCP server tools for parking reservations
"""

from fastmcp import FastMCP

mcp = FastMCP("Parking Reservations")


@mcp.tool  # type: ignore[misc]
def create_reservation_file(user_name: str, user_surname: str, plate_number: str, period: str) -> dict[str, str]:
    """Create reservation file"""
    print(f"Approved reservation for {user_name} {user_surname}: {plate_number} {period}")
    return {"status": "created"}


app = mcp.http_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
