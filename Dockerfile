# API image: FastAPI + the RAG platform (Phase 7).
#   docker compose up        (see docker-compose.yml; builds this and the UI image)
#
# Layers are ordered so that a code change does not reinstall the dependencies: first only
# pyproject.toml + uv.lock (cached as long as they don't change), then the source.
# .NET analogy: like a multi-stage Dockerfile that runs `dotnet restore` before copying the code.

FROM python:3.12-slim

# uv (the package manager) copied from its official image; pinned to the version used locally.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# 1) Dependencies only (no dev tools, no project code yet) -> cached layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) The project: code, active settings, sample corpus and evaluation data.
COPY README.md rag_config.json assistant.json model_prices.json ./
COPY src ./src
COPY data/sample_corpus ./data/sample_corpus
COPY data/eval ./data/eval
RUN uv sync --frozen --no-dev

# Runtime data (index, uploads) lives in volumes, see docker-compose.yml. Secrets come from
# the environment (env_file), never from the image.

# Run as a normal user, not root (security plan B4): if someone ever broke into the app, they
# couldn't change the image's files or install anything. The code stays root-owned and
# read-only for this user; it writes only to the mounted data folders. HOME=/tmp gives
# libraries a writable place for caches.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
ENV HOME=/tmp
USER app

EXPOSE 8000
# The virtual environment's own `rag` command (uv itself is not needed at runtime).
CMD ["/app/.venv/bin/rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
