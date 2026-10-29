FROM ubuntu:24.04 AS runtime-base

ARG DEBIAN_FRONTEND=noninteractive


# Pre-accept the Microsoft EULA for ttf-mscorefonts-installer
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

# ── Runtime-only system dependencies    ──────────────────────────────────────
# (no build tools, no -dev headers – those belong in the builder stage)
RUN apt-get update \
  && apt-get -y install --no-install-recommends \
    x11-xserver-utils libglut3.12  libglu1-mesa \
    libgirepository-2.0-0 libxkbcommon0 libegl1 \
    libfontconfig1 libxcb-cursor-dev libxkbcommon-x11-0 \
    libqt5core5t64 xcb libx11-xcb1 libxcb-icccm4 \
    libxcb-keysyms1 libxcb-shape0 dbus \
    libportaudio2 \
    libasound2t64 pulseaudio libpulse0 \
    libavcodec60 libpostproc57 libavdevice60 libavfilter9 libavformat60 libsdl2-2.0-0 \
    ttf-mscorefonts-installer fonts-ubuntu fonts-liberation cabextract fontconfig\
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Builder stage – compiles from source + installs Python deps ────────────
FROM runtime-base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-linux-gnu
ENV GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0/

RUN uv venv --python 3.12 --managed-python /src/.venv --system-site-packages

RUN apt-get update \
  && apt-get -y install --no-install-recommends \
  build-essential pkg-config \
  libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev libsdl2-dev libpostproc-dev git ca-certificates

# ── Install MultiTaskBattery and its deps ─────────────────────────────
COPY ./ /src/mtb

RUN . /src/.venv/bin/activate \
    && echo 'scipy==1.18' > /tmp/constraints.txt \
    && uv pip install --no-cache-dir /src/mtb \
       --constraints /tmp/constraints.txt

FROM runtime-base AS runtime

WORKDIR /src/mtb

COPY docker/pulse-client.conf /etc/pulse/client.conf
COPY --from=builder /src /src
COPY --from=builder /root/.local/share/uv /root/.local/share/uv

RUN fc-cache -f -v

ENV PATH=/src/venv/bin:$PATH
