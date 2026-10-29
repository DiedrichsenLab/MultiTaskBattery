Installation
============

There are two ways to run MTB: install it with Python (any OS), or use the
ready-made Docker container (Linux).

Step 1: Clone the Repository
----------------------------

Using Git::

    git clone https://github.com/diedrichsenlab/MultiTaskBattery.git
    cd MultiTaskBattery

Or use `GitHub Desktop <https://desktop.github.com/>`_.

Step 2: Install Python (3.10 - 3.12)
------------------------------------

MTB runs on **Python 3.10, 3.11 or 3.12** (PsychoPy does not support newer
versions yet). We recommend **3.12**.

Option A: Using pyenv (Recommended on macOS/Linux)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install pyenv::

    brew update
    brew install pyenv

Configure your shell::

    echo 'if command -v pyenv 1>/dev/null 2>&1; then eval "$(pyenv init -)"; fi' >> ~/.bash_profile
    source ~/.bash_profile

Install Python::

    pyenv install 3.12
    pyenv global 3.12

Option B: Using system Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure ``python --version`` reports 3.10, 3.11 or 3.12.

Step 3: Create a Virtual Environment
------------------------------------

::

    python -m venv mtb-env
    source mtb-env/bin/activate    # On Windows: mtb-env\Scripts\activate

Step 4: Install MultiTaskBattery
--------------------------------
Make sure you upgrade pip first, then install the package (this also
pulls in all dependencies).

::

    pip install --upgrade pip
    pip install -e .

The ``-e`` means Python uses your cloned folder directly, so after
``git pull`` the new code is used right away, without installing again.

Step 5: Download the Stimuli
----------------------------
The task stimuli (images, audio, video and text) are not in the repository; they are
hosted on `Zenodo <https://doi.org/10.5281/zenodo.23749588>`_, one zip per
task. Download them once, while online::

    mtb fetch                       # stimuli of every task (about 1 GB)
    mtb fetch reading movie         # only the tasks you use

They are saved in your computer's cache folder (e.g.
``~/.cache/MultiTaskBattery`` on Linux), where MTB finds them from any
experiment folder. To keep them somewhere else, set the ``MTB_STIMULI``
environment variable to that folder, both when fetching and when running. A
task whose stimuli are missing stops with a message saying which
``mtb fetch`` to run.

Alternative: Docker (Linux)
---------------------------

The container has Python, PsychoPy and MTB preinstalled, so no Python setup
is needed. You only need ``docker/run_docker.sh`` from the
repository (clone it, or download that one file). Install `Docker <https://docs.docker.com/engine/install/>`_,
then pull the container::

    docker pull ghcr.io/diedrichsenlab/multitaskbattery:latest

For a study, use a fixed release instead of ``latest`` (e.g.
``multitaskbattery:1.2.1``), so every session and site runs exactly the same
version.

``docker/run_docker.sh`` in the repository starts the container with access to
the screen, sound and the scanner keyboard, and runs a Python script in it::

    bash docker/run_docker.sh <script.py>

It uses ``latest`` by default. To use a fixed release, set ``MTB_VERSION``::

    MTB_VERSION=1.2.1 bash docker/run_docker.sh <script.py>

The stimuli are not in the container. ``run_docker.sh`` gives the container
the same stimulus folder on your computer every time, so fetch them once,
while online::

    bash docker/run_docker.sh mtb fetch reading movie
