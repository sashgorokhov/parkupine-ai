from typing import AsyncGenerator
from unittest import mock

import pytest
from fastapi_openai_compat import ChatRequest

from parkupine.agent import manual_chat_completion


@pytest.mark.parametrize("endpoint", ["/models", "/v1/models"])
def test_models(endpoint, client, app_settings):
    response = client.get(endpoint)
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == app_settings.model_id
    assert response.json()["data"][0]["name"] == app_settings.model_name
    assert response.json()["data"][0]["owned_by"] == app_settings.model_owner


@pytest.mark.parametrize("endpoint", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_unauthorized_bad_bearer(endpoint, client, set_auth_headers):
    client.headers["Authorization"] = f"Bearer wrong"

    response = client.post(endpoint)
    assert response.status_code == 401


@pytest.mark.parametrize("endpoint", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_unauthorized_no_bearer(endpoint, client, set_auth_headers):
    client.headers.pop("Authorization")

    response = client.post(endpoint)
    assert response.status_code == 401


@pytest.mark.parametrize("endpoint", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_unauthorized_no_user(endpoint, client, set_auth_headers):
    client.headers.pop("x-openwebui-user-id")

    response = client.post(endpoint)
    assert response.status_code == 422


def make_async_generator(*return_value) -> AsyncGenerator:
    async def generator() -> AsyncGenerator:
        for rv in return_value:
            yield rv

    return generator()


@pytest.mark.parametrize("endpoint", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_submits_chat_request(endpoint, client, set_auth_headers, chat_request):
    chat_completion = manual_chat_completion(
        message="test",
        model="test",
        stream=False,
    )

    with mock.patch("parkupine.server.submit_chat_request") as m:
        m.return_value = make_async_generator(chat_completion)

        response = client.post(endpoint, json=chat_request.model_dump(mode="json"))

    assert response.status_code == 200, response.content
    assert response.json() == chat_completion.model_dump(mode="json")
