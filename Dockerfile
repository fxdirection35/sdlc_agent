# Multi-stage image: the SPA builds in stage 1, the service installs in
# stage 2 and serves it same-origin. A deployment therefore never depends
# on a developer's local npm state or on frontend/dist existing.

# ---------------------------------------------------------------------------
# Stage 1: build the SPA on the same Node line the team develops on.
# ---------------------------------------------------------------------------
FROM node:24-alpine AS frontend
WORKDIR /build
# Lockfile first, so dependency layers cache independently of source edits.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build # tsc --noEmit + vite build -> dist/

# ---------------------------------------------------------------------------
# Stage 2: the service.
#
# Installs the sandbox extra because the deployed app composes itself at
# import time (service.main -> executor_from_env -> sdlc_agent.sandbox) - a
# core-only image would crash at boot, not at first use. psycopg[binary]
# ships wheels, so slim needs no system libpq.
#
# Installs git because the code-flow nodes (checkout/apply/deliver) drive it
# through subprocess: slim ships without it, and its absence is a boot-green,
# first-run-red image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS api
# pip mirror as a build arg, defaulting to the one this team needs
# (CLAUDE.md): files.pythonhosted.org times out from CN networks. Override
# with PIP_INDEX_URL=https://pypi.org/simple when building elsewhere.
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# Same story for apt: deb.debian.org is unreachable from some CN networks.
# Override with APT_MIRROR=deb.debian.org when building elsewhere.
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
RUN sed -i "s/deb.debian.org/${APT_MIRROR}/g; s/security.debian.org/${APT_MIRROR}/g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# README.md is referenced by pyproject.toml; hatchling fails without it.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" ".[sandbox]"

# The SPA from stage 1, served by the service itself (SDLC_STATIC_DIR).
COPY --from=frontend /build/dist/ /app/static/
# config.yaml is NOT baked: it is mounted read-only at runtime, so a config
# edit is a restart, not a rebuild, and no key name ever enters an image layer.
ENV SDLC_STATIC_DIR=/app/static \
    SDLC_CONFIG_PATH=/app/config/config.yaml

RUN useradd --create-home --shell /usr/sbin/nologin sdlc
# Per-run repository clones (SDLC_WORKSPACES_DIR). Pre-created with sdlc
# ownership because a named volume's FIRST mount copies this directory's
# ownership - without it the non-root service cannot write into the volume
# and every run escalates at checkout.
RUN mkdir -p /app/workspaces && chown sdlc:sdlc /app/workspaces
ENV SDLC_WORKSPACES_DIR=/app/workspaces
# Nothing in the service writes inside /app; run as a plain user anyway.
USER sdlc

# No curl on slim; health is a stdlib one-liner against the service's own
# /api/health (which is only reachable once config + credentials loaded).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]

EXPOSE 8000
# 0.0.0.0 on purpose: the container-default 127.0.0.1 leaves the published
# port forwarding to nothing - a green container nobody can reach.
CMD ["uvicorn", "sdlc_agent.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
