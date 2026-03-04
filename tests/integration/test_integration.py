from threading import Thread

import pytest
from fastapi_openai_compat import ChatCompletion
from langchain_core.messages import AIMessage, AIMessageChunk


@pytest.fixture(autouse=True)
def run_worker(worker):
    t = Thread(target=worker.start, daemon=True)
    t.start()
    yield
    worker.stop()
    t.join()


@pytest.mark.parametrize("endpoint", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_integration(endpoint, client, set_auth_headers, chat_request, model):
    model.mock._generate.side_effect = [AIMessage("test", response_metadata={"finish_reason": "stop"})]

    response = client.post(endpoint, json=chat_request.model_dump(mode="json"))

    chat_completion = ChatCompletion.model_validate(response.json())
    assert chat_completion.choices[0].message.content == "test"


@pytest.mark.parametrize("endpoint", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_integration_streaming(endpoint, client, set_auth_headers, chat_request, model):
    model.mock._generate.side_effect = [
        [
            AIMessageChunk(content="1"),
            AIMessageChunk(content="2"),
            AIMessageChunk(content="3", chunk_position="last", response_metadata={"finish_reason": "stop"}),
        ]
    ]
    chat_request.stream = True

    response = client.post(endpoint, json=chat_request.model_dump(mode="json"))

    serialized_messages = [m.lstrip("data: ") for m in response.content.decode("utf8").split("\n\n") if m]
    messages: list[ChatCompletion] = list(map(ChatCompletion.model_validate_json, serialized_messages))

    assert len(messages) == 3
    assert messages[0].choices[0].delta.content == "1"
    assert messages[1].choices[0].delta.content == "2"
    assert messages[2].choices[0].delta.content == "3"
