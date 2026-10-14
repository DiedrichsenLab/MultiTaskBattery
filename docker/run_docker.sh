docker run \
  -v "./stimuli:/src/.venv/lib/python3.14/site-packages/stimuli" \
  --device=/dev/dri \
  --device=/dev/snd \
  -v "/run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${XAUTHORITY}:${XAUTHORITY}" \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -e DISPLAY \
  -e XAUTHORITY \
  -e PULSE_SERVER \
  -it docker pull ghcr.io/digital-brain-project/multitaskbattery:latest /src/.venv/bin/python \
  $@
