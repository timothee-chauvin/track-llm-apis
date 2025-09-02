# Reference: https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile

FROM python:3.11-slim@sha256:1d6131b5d479888b43200645e03a78443c7157efbdb730e6b48129740727c312

COPY --from=ghcr.io/astral-sh/uv:0.8.13@sha256:4de5495181a281bc744845b9579acf7b221d6791f99bcc211b9ec13f417c2853 /uv /uvx /bin/

ENV TZ=UTC \
    LANGUAGE=en_US:en \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Dependencies
RUN apt-get update && apt-get install -y \
    # for uv dependencies based on git repositories
    git \  
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR=/app

RUN mkdir -p $CODE_DIR && chmod -R 777 $CODE_DIR
WORKDIR $CODE_DIR

ENV UV_PROJECT_ENVIRONMENT=/venv \
    UV_CACHE_DIR=/uv_cache \
    # Copy from the cache instead of linking since it's a mounted volume
    UV_LINK_MODE=copy
    
RUN mkdir -p $UV_CACHE_DIR && chmod -R 777 $UV_CACHE_DIR

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . .
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --locked --no-dev

RUN uv build

RUN chmod -R 777 $UV_CACHE_DIR $UV_PROJECT_ENVIRONMENT

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
