# Usage: bash docker/run_docker.sh <script.py>
# Pin a release with: MTB_VERSION=1.2.0 bash docker/run_docker.sh <script.py>
MTB_VERSION=${MTB_VERSION:-latest}

# Only mount the X authority file if it is set (it is unset on some setups)
XAUTH_ARGS=()
if [ -n "${XAUTHORITY}" ]; then
  XAUTH_ARGS=(-v "${XAUTHORITY}:${XAUTHORITY}" -e XAUTHORITY)
fi

docker run \
  -v "$PWD:/exp" -w /exp \
  --device=/dev/dri \
  --device=/dev/snd \
  -v "/run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "${XAUTH_ARGS[@]}" \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -e DISPLAY \
  -e PULSE_SERVER \
  -it ghcr.io/diedrichsenlab/multitaskbattery:${MTB_VERSION} /src/.venv/bin/python \
  "$@"
