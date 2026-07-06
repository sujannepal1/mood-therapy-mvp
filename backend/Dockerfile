FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifest first — Docker layer cache: deps only rebuild when pyproject.toml changes
COPY pyproject.toml .

# Install dependencies (skip installing the project itself — we run src directly)
RUN uv sync --no-install-project

# Copy all source code
COPY . .

# entrypoint: trains models on first run if missing, then execs the CMD
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
