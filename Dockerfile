# syntax=docker/dockerfile:1
#
# Single-server image: FastAPI + built React UI (README production mode).
#
# Build:
#   docker build -t wq-alpha-miner .
#
# Run (mount secrets, config, and persistent db):
#   docker run --rm -p 8000:8000 \
#     --env-file .env \
#     -v "$PWD/config.yaml:/app/config.yaml:ro" \
#     -v "$PWD/db:/app/db" \
#     wq-alpha-miner
#
# Entrypoint seeds db/*.parquet via scripts/init_wiki.py when missing.
# ── frontend ──────────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── python app ────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

# libgomp: required by scikit-learn wheels at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependency layer (cached unless lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Project install (editable so Path(__file__) resolves to /app/…)
COPY wq_alpha_miner ./wq_alpha_miner
COPY scripts ./scripts
COPY config.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

COPY --from=frontend /frontend/dist ./frontend/dist

RUN mkdir -p db \
    && chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "wq_alpha_miner.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
