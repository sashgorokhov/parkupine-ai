"""
FastApi entrypoint, routes, and handler functions.

Implements OpenAI-compatible API routes that are used by OpenWebUI as "imposter" openai model.

Endpoints:
- /models: Returns list of supported models (just one - Parkupine)
- /chat/completions: Return AI response to user message

This module uses fastapi_openai_compat functions to outsource boring streaming and modeling code.
"""

import logging
import time
from typing import Any, Annotated

from fastapi import FastAPI, HTTPException, Header
from fastapi.openapi.utils import get_openapi
from fastapi_openai_compat import (
    ModelsResponse,
    ModelObject,
    ChatRequest,
    ChatCompletion,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from parkupine.auth import UserDep
from parkupine.context import AppContext
from parkupine.dependencies import AppSettingsDep, AppContextDep
from parkupine.settings import APP_TITLE, APP_DESCRIPTION, DEBUG
from parkupine.worker import submit_chat_request

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(name)s] %(message)s")

logger = logging.getLogger(__name__)


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, lifespan=AppContext, debug=DEBUG)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/models")
@app.get("/models")
async def models(settings: AppSettingsDep, user: UserDep) -> ModelsResponse:
    """
    Returns a list of available models (deployed pipelines) in OpenAI-compatible format.
    Each model object contains an `id` that can be used as the `model` field
    in chat completion requests.

    References:
    - [OpenAI Models API](https://platform.openai.com/docs/api-reference/models/list)
    - [Ollama OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md)
    """
    if user.is_user:
        m = [
            ModelObject(
                id=settings.model_id,
                name=settings.model_name,
                object="model",
                created=int(time.time()),
                owned_by=settings.model_owner,
            )
        ]
    elif user.is_admin:
        m = [
            ModelObject(
                id=settings.admin_model_id,
                name=settings.admin_model_name,
                object="model",
                created=int(time.time()),
                owned_by=settings.model_owner,
            )
        ]
    else:
        m = []

    return ModelsResponse(
        data=m,
        object="list",
    )


class OpenwebuiChatHeaders(BaseModel):
    """
    Headers provided by OpenWebUi

    Reference: https://docs.openwebui.com/reference/env-configuration/#enable_forward_user_info_headers
    """

    # x_openwebui_message_id: str = Field(alias="x-openwebui-message-id")
    x_openwebui_chat_id: str = Field(alias="x-openwebui-chat-id")


@app.post("/v1/chat/completions", response_model=ChatCompletion)
@app.post("/chat/completions", response_model=ChatCompletion)
async def chat_completions(
    chat_request: ChatRequest,
    context: AppContextDep,
    user: UserDep,
    chat_headers: Annotated[OpenwebuiChatHeaders, Header()],
) -> ChatCompletion | StreamingResponse:
    """
    Generates a chat completion for the given conversation in OpenAI-compatible format.
    **Non-streaming** (`stream: false`, default): returns a single JSON `ChatCompletion` object.
    **Streaming** (`stream: true`): returns a stream of server-sent events (SSE),
    each containing a `ChatCompletion` chunk with incremental content in `choices[].delta`.

    References:
    - [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
    """
    # Should use permissions pattern dependencies instead
    if not user.is_admin and chat_request.model == context.settings.admin_model_id:
        raise HTTPException(status_code=403, detail="Not an admin")

    try:
        logger.debug(f"Received chat request {chat_request} from {user}")
        # This will send chat request into redis queue, and wait for stream of tokens and return them here
        result = submit_chat_request(
            redis=context.redis, chat_id=chat_headers.x_openwebui_chat_id, chat_request=chat_request, user=user
        )

        if chat_request.stream:
            logger.debug("LLM response is a stream")
            return StreamingResponse(result, media_type="text/event-stream")
        else:
            response = await anext(result)
            logger.debug(f"LLM response: {response}")
            return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline execution error")
        raise HTTPException(status_code=500) from exc


def custom_openapi() -> dict[str, Any]:
    """
    This function is a patch for https://github.com/fastapi/fastapi/pull/12481 where openapi spec
    is incorrectly generated for routes with multiple pydantic models in its dependencies.
    I only need this for convenience of testing routes through swagger ui.

    This function will replace parameter spec for models with flattened parameter list, as expected.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=APP_TITLE,
        version="0.1",
        summary=APP_DESCRIPTION,
        routes=app.routes,
    )

    # Iterate over offending paths
    for path in ["/chat/completions", "/v1/chat/completions"]:
        parameters = openapi_schema["paths"][path]["post"]["parameters"]

        # Go through each defined route parameter, if they are header related and have complex schema,
        # remove parameter from parameter list completely
        for parameter in parameters[:]:
            if parameter["in"] == "header" and "$ref" in parameter["schema"]:
                parameters.remove(parameter)

                # ref looks like #/components/schemas/PydanticModelName
                ref = parameter["schema"]["$ref"].split("/")[-1]
                schema = openapi_schema["components"]["schemas"][ref]

                # Iterate through each schema property and create separate parameter for each
                for property_name, property in schema["properties"].items():
                    parameters.append(
                        {
                            "in": "header",
                            "name": property_name,
                            "required": property_name in schema["required"],
                            "schema": property,
                        }
                    )

                    logger.debug(f"Patched openapi schema @ {path}: replaced {parameter} with {parameters[-1]}")

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
