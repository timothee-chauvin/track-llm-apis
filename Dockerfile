# Reference: https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile

FROM python:3.11-slim@sha256:1d6131b5d479888b43200645e03a78443c7157efbdb730e6b48129740727c312

COPY --from=ghcr.io/astral-sh/uv:0.8.13@sha256:4de5495181a281bc744845b9579acf7b221d6791f99bcc211b9ec13f417c2853 /uv /uvx /bin/

# Dependencies
RUN apt-get update && apt-get install -y \
    # for uv dependencies based on git repositories
    git \  
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user: "tluser" for "track LLMs user"
ENV UID=1000
RUN useradd --uid $UID --create-home --user-group tluser
USER tluser
ENV HOME=/home/tluser
WORKDIR $HOME/trackllm

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

ENV UV_CACHE_DIR=$HOME/.cache/uv

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=$UV_CACHE_DIR,uid=$UID,gid=$UID \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY --chown=$UID:$UID . .
RUN --mount=type=cache,target=$UV_CACHE_DIR,uid=$UID,gid=$UID \
    uv sync --locked --no-dev

RUN uv build

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
