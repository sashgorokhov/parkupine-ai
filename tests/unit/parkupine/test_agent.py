from langchain_core.messages import AIMessage


def test_agent_handle_chat_request(chat_request, user, agent, model):
    model.mock._generate.return_value = AIMessage("test")

    completions = list(agent.handle_chat_request(chat_request=chat_request, user=user, chat_id="test"))

    assert len(completions) == 1
    assert completions[0].choices[0].message.content == "test"
