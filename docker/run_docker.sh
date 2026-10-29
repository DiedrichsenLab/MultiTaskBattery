# Usage: bash docker/run_docker.sh <script.py>
# Uses the MTB release below by default; pick another image with: MTB_VERSION=latest bash docker/run_docker.sh <script.py>
# Fetch task stimuli once (online): bash docker/run_docker.sh mtb fetch <task> ...
MTB_VERSION=${MTB_VERSION:-1.3.0}   # update at each release

# `run_docker.sh mtb ...` runs the mtb command; anything else runs python
CMD=(/src/.venv/bin/python)
if [ "$1" = "mtb" ]; then CMD=(/src/.venv/bin/mtb); shift; fi

# Host folder for task stimuli (the same one a pip install uses), seen as /stimuli in the container
STIM_DIR=${MTB_STIMULI:-$HOME/.cache/MultiTaskBattery/stimuli}
mkdir -p "$STIM_DIR"

# Only mount the X authority file if it is set (it is unset on some setups)
XAUTH_ARGS=()
if [ -n "${XAUTHORITY}" ]; then
  XAUTH_ARGS=(-v "${XAUTHORITY}:${XAUTHORITY}" -e XAUTHORITY)
fi

docker run \
  -v "$PWD:/exp" -w /exp \
  -v "$STIM_DIR:/stimuli" -e MTB_STIMULI=/stimuli \
  --device=/dev/dri \
  --device=/dev/snd \
  -v "/run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "${XAUTH_ARGS[@]}" \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -e DISPLAY \
  -e PULSE_SERVER \
  -it ghcr.io/diedrichsenlab/multitaskbattery:${MTB_VERSION} "${CMD[@]}" \
  "$@"
