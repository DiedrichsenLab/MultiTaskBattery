FROM ubuntu:26.04 AS runtime-base

ARG DEBIAN_FRONTEND=noninteractive

# ── Runtime-only system dependencies    ──────────────────────────────────────
# (no build tools, no -dev headers – those belong in the builder stage)
RUN apt-get update \
  && apt-get -y install --no-install-recommends \
    x11-xserver-utils libglut3.12  libglu1-mesa \
    libgirepository-2.0-0 libxkbcommon0 libegl1 \
    libfontconfig1 libxcb-cursor-dev libxkbcommon-x11-0 \
    libqt5core5t64 xcb libx11-xcb1 libxcb-icccm4 \
    libxcb-keysyms1 libxcb-shape0 dbus

# ── Builder stage – compiles from source + installs Python deps ────────────
FROM runtime-base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-linux-gnu
ENV GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0/

RUN uv venv /src/.venv --system-site-packages


RUN apt-get update \
  && apt-get -y install --no-install-recommends \
  build-essential pkg-config

# ── Install MultiTaskBattery and its deps ─────────────────────────────
COPY ./ /src/mtb

RUN . /src/.venv/bin/activate \
    && echo 'ffpyplayer' >> /tmp/excludes.txt \
    && echo 'scipy==1.18' > /tmp/constraints.txt \
    && uv pip install --no-cache-dir /src/mtb \
       --find-links https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-24.04/ \
       --only-binary wxPython \
       --constraints /tmp/constraints.txt \
       --excludes /tmp/excludes.txt
