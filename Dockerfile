FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app
EXPOSE 8000

# For health checks and installing from git
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache pip install poetry==2.3.2

COPY pyproject.toml poetry.lock README.md ./

RUN --mount=type=cache,target=/root/.cache poetry install --no-interaction --no-ansi --no-root --only main

COPY . ./

# Run uvicorn server
CMD ["uvicorn", "parkupine.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
