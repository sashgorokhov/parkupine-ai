"""
Core agent logic
"""

from fastapi_openai_compat import ChatRequest, CompletionResult

from parkupine.auth import BaseUser


async def handle_chat_request(chat_request: ChatRequest, chat_id: str, message_id: str, user: BaseUser) -> CompletionResult:
    return f'howdy, {user.name}!'
