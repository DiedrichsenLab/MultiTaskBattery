"""One-command dependency install for MultiTaskBattery, on any OS.

    python install.py

On Linux this detects your distribution and points pip at the matching
prebuilt wxPython wheels (wxPython has none on PyPI for Linux, so a plain
pip install would try to compile it from source and fail). On Windows and
macOS it is exactly `pip install -r requirements.txt`.

`python install.py --dry-run` shows the command without running it.
"""
import os
import subprocess
import sys
from pathlib import Path

WHEEL_BASE = "https://extras.wxpython.org/wxPython4/extras/linux/gtk3"


def linux_distro():
    """Return e.g. 'ubuntu-24.04' from /etc/os-release, or None."""
    info = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v.strip().strip('"')
    except OSError:
        return None
    if "ID" in info and "VERSION_ID" in info:
        return f"{info['ID']}-{info['VERSION_ID']}"
    return None


def main():
    env = os.environ.copy()
    if sys.platform.startswith("linux"):
        distro = linux_distro()
        if distro is None:
            sys.exit("Could not detect the Linux distribution from /etc/os-release.\n"
                     f"Pick your release at {WHEEL_BASE}/ and run:\n"
                     "  MTB_WX_DISTRO=<release> pip install -r requirements.txt")
        env["MTB_WX_DISTRO"] = distro
        print(f"Detected {distro}; using prebuilt wxPython wheels from\n"
              f"  {WHEEL_BASE}/{distro}/\n")

    cmd = [sys.executable, "-m", "pip", "install", "-r",
           str(Path(__file__).parent / "requirements.txt")]
    print("Running:", " ".join(cmd))
    if "--dry-run" in sys.argv:
        return
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
