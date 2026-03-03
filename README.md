<h1 align="center">parkupine</h1>
<p align="center">
  <img src="assets/logo.png" height="128" />
</p>

Your favourite parking reservation assistant.

The goal of the project is to develop an intelligent chatbot that can interact with users, provide information about parking spaces, handle the reservation process, and involve a human administrator for confirmation ("human-in-the-loop"). The project will be divided into 4 stages, with each stage implementing a specific functionality.

Some of the used software:
- [open-webui](https://github.com/open-webui/open-webui)
- [pgvector](https://github.com/pgvector/pgvector)
- [sqlmodel](https://sqlmodel.tiangolo.com/)
- [fastapi](https://fastapi.tiangolo.com/)
- [langchain](https://docs.langchain.com/oss/python/langchain/overview)
- [valkey](https://valkey.io/)

Under the hood, this is a Open WebUI frontend with FastApi OpenAI-compatible backend. Agent logic is executed asynchronously
on separate workers, with communication through redis queue and channels.

## Features

- Streaming tokens!
- OpenAI-compatible API, plug and play with any LLM tool!
- Asynchronous LLM invocation with redis workers! Scalability!
- Fully dockerized and container native!
- 86% artisanal hand-crafted code!

## Usage

Prerequisites:
- OpenAI api key
- docker and docker compose

1. Copy private.env.example into private.env and set PARKUPINE_OPENAI_API_KEY to your OpenAI api key
2. Run `docker-compose up -d --wait`. This will start all services.
3. Run `docker-compose exec worker python -m parkupine.tables`. This will pre-populate DB with some data. This step is idempotent.
4. Access Open WebUI at http://localhost:8080/ and signup with any name, email and password


Cleanup:
```shell
docker-compose.exe down -v --remove-orphans
```

Viewing server and worker logs:
```shell
docker-compose logs -f parkupine worker
```

Rebuild server and worker:
```shell
docker-compose.exe up -d --build --force-recreate parkupine worker
```

Swagger UI: http://localhost:8000/docs

## Implementation details

"Customer" facing frontend is implemented through OpenWebUi. It handles user registration, authentication and
interaction with underlying agent logic. It is highly configurable and you can find 100+ environment
variables in this [.env](common.env) file.

OpenWebUI communicates with FastApi [server](parkupine/server.py) that mimics OpenAI interface. This server has just two
endpoints, one of which is chat completions.

This endpoint authenticates user through bearer api token and collects all context information, and creates
`ChatWorkItem` and puts it into `chat_requests` queue. It then subscribes to request-scoped channel in redis
and waits for LLM tokens, which are streamed back to client.

In parallel, there is a [Worker](parkupine/worker.py) instance in another process that polls `chat_requests` queue for new `ChatWorkItem`s.
Once appeared, it passes all context to [Agent](parkupine/agent.py) `.handle_chat_request` method. It runs graph
and returns LLM tokens, which are published into request-scoped channel mentioned earlier.

General data flow can be described as:

User -> Open WebUI -> fastapi `/v1/chat/completions` -> redis queue -> worker process -> Agent class
-> LangGraph `.invoke`

## Tradeoffs and cut corners

- User management is very rudimentary and not suitable for production. Single token authenticates all users.
- Relies on openwebui headers for user identification which limits adaptability.

---

General Requirements:
Programming Language: Python.
Frameworks: LangChain, LangGraph.
Architecture: Based on Retrieval-Augmented Generation (RAG).
Vector database: Recommended options include Milvus, Pinecone, or Weaviate,
General Features:
The chatbot provides information (general information, working hours, prices, availability of parking spaces, location).
The reservation process is based on interactive collection of user data, including name, surname, car number, and reservation period.
The system should prevent exposure of sensitive data (e.g., private information stored in the vector database).
Evaluation of system performance (e.g., request latency, information retrieval accuracy).

Providing the result:
for each task, please provide a link to your GitHub or EPAM GitLab repository in the answer field
you can earn extra points if you provide the following artifacts:
- a PowerPoint presentation explaining how the solution works, including relevant screenshots
- a README file with clear project documentation (setup, usage, structure, etc.)
- Automated test cases are implemented using pytest or unittest  (at least 2 tests per module)
- CI/CD automation and/or Infrastructure as Code (e.g., Terraform)
- If the code is poor quality, or too basic to be practical, and includes critical errors, the grade may be reduced


---

Tasks:
1. Implement the basic architecture of the chatbot using Retrieval-Augmented Generation to interact with users.
2. Integrate a vector database for storing information.
(Optional) Solution can be improved by splitting source data into 2 data types: dynamic data (e.g. space availability, working hours, prices) and  static data (e.g. general information, parking details, location, booking process).  Store static data in vector database but dynamic data in SQL database.

3. Implement interactive features:
Provide information to users.
Collect user inputs for reservations.

4. Implement guard rails mechanism. Add a filtering to prevent exposure of sensitive data (e.g., using pre-trained NLP models for text analysis).
5. Perform evaluation of the RAG system: Performance testing. Response accuracy measurement (e.g., using metrics like Recall@K and Precision).


    Outcome:
    Working chatbot capable of providing basic information and interacting with users.
    Data protection functionality.
    Evaluation report on system performance.

---

Tasks:
Create the second agent using LangChain to interact with the administrator.
Chat bot should be able to send a reservation request to administrator and get confirm/refuse response from him.  (e.g. via email server, messenger, rest api ).
Organize the integration with the first agent so that the reservation request is escalated to the human administrator after collecting details from the user.

Key Features:
Generating and sending reservation confirmation requests to the administrator.
Receiving responses from the administrator.
Maintaining communication between the first and second agents.

    Outcome:
    Automated system that connects an administrator for reservation approval.

---

Tasks:
Use any open-source MCP server that provides functionality to write data to file.  Alternatively, develop a simple MCP server using Python + FastApi to process confirmed reservations.
In case of MCP server is not possible to implement, use tool/function call for writing dada into file.

Once the administrator (second agent) approves the reservation, the server should write the reservation details to a text file.
File entry format: Name | Car Number | Reservation Period | Approval Time.

Ensure the server is secure and resistant to unauthorized access, while ensuring reliable service.

    Outcome:
    A fully functional MCP server integrated with the previous agents.
    The server processes reservation data and saves it in storage.

---

Tasks:
Implement orchestration of all components using LangGraph.
Ensure complete integration of all stages:
The chatbot (RAG agent) interacts with users.
The system escalates reservation requests to the administrator via a human-in-the-loop agent (second agent).
The MCP server processes data after confirmation.

Implement the workflow logic for the entire pipeline:
Example graph structure:
Node for user interaction (context of RAG and chatbot).
Node for administrator approval.
Node for data recording.

Conduct testing of the entire system workflow.

    Outcome:
    A unified system where all components seamlessly interact with each other.
    Stable operation of the entire pipeline.

Additional Details:
System Testing:
Conduct load tests to evaluate the performance of each component:
Chatbot in interactive dialogue mode.
Administrator confirmation functionality.
MCP server recording and storage process.

Perform integration testing of all steps during orchestration.

Documentation:
Prepare documentation for system usage:
Architecture description.
Agent and server logic.
Setup and deployment guidelines.
