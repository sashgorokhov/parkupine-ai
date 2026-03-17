import operator

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from openevals import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_HELPFULNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT
from sqlmodel import select

from parkupine.agent import USER_SYSTEM_PROMPT
from parkupine.tables import populate_data, ParkingGarage, ParkingSpace

pytestmark = pytest.mark.vcr


@pytest.fixture(autouse=True)
def populate_db(engine):
    populate_data(engine)


@pytest.fixture()
def correctness_evaluator(openai_api_key):
    return create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model="openai:o3-mini",
    )


@pytest.fixture()
def helpfulness_evaluator(openai_api_key):
    return create_llm_as_judge(
        prompt=RAG_HELPFULNESS_PROMPT,
        feedback_key="helpfulness",
        model="openai:o3-mini",
    )


@pytest.fixture()
def relevance_evaluator(openai_api_key):
    return create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        feedback_key="retrieval_relevance",
        model="openai:o3-mini",
    )


# All BaseMessage instances must have their id hardcoded for VCR reproducibility


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            [HumanMessage("What can you do?", id="1")],
            AIMessage("I am parkupine, parking reservation assistant. I can help with reservations."),
        ),
        (
            [HumanMessage("I need to park somewhere in downtown", id="1")],
            AIMessage("Sure, we have Green Garage available in downtown"),
        ),
        (
            [HumanMessage("Where is Green Garage located?", id="1")],
            AIMessage("Green Garage is located at 123 Main St, near business center"),
        ),
    ],
)
def test_agent_evaluation(
    inputs,
    expected,
    agent,
    correctness_evaluator,
    populate_db,
    helpfulness_evaluator,
    subtests,
    relevance_evaluator,
    db_session,
):
    result = agent._graph.invoke(
        input={"messages": inputs},
        config=RunnableConfig(
            configurable={
                "thread_id": "test",
            }
        ),
    )
    outputs = result["messages"][-1]
    outputs.id = "1"

    with subtests.test(eval="correctness"):
        evaluation = correctness_evaluator(inputs=inputs, outputs=outputs, reference_outputs=expected)

        assert evaluation["score"], evaluation["comment"]

    with subtests.test(eval="helpfulness"):
        evaluation = helpfulness_evaluator(inputs=inputs, outputs=outputs, reference_outputs=expected)

        assert evaluation["score"], evaluation["comment"]

    garages = list(map(operator.methodcaller("model_dump"), db_session.exec(select(ParkingGarage)).all()))
    spaces = list(map(operator.methodcaller("model_dump"), db_session.exec(select(ParkingSpace)).all()))

    context = {"documents": garages + spaces + [USER_SYSTEM_PROMPT]}

    with subtests.test(eval="relevance"):
        evaluation = relevance_evaluator(inputs=inputs, outputs=outputs, reference_outputs=expected, context=context)

        assert evaluation["score"], evaluation["comment"]
