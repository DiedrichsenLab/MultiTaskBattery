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
    python install.py

``install.py`` runs ``pip install -e .``. On Linux it additionally detects your
distribution and fetches a prebuilt wxPython wheel (wxPython has no Linux
wheels on PyPI, so a plain pip install would fail trying to compile it).
On Windows and macOS it is equivalent to ``pip install -e .``, which you can
run directly.

The ``-e`` (editable) install points Python at the cloned folder, so
``import MultiTaskBattery`` works from anywhere and the bundled ``stimuli/``
folder next to the code is found. Do not use a plain ``pip install .``: it
copies only the code, not the stimuli. Pulling updates with ``git pull`` takes
effect without reinstalling.

Alternative: Docker (Linux)
---------------------------

The container has Python, PsychoPy, MTB and its stimuli preinstalled, so no
Python setup is needed. You only need ``docker/run_docker.sh`` from the
repository (clone it, or download that one file). Install `Docker <https://docs.docker.com/engine/install/>`_,
then pull the container::

    docker pull ghcr.io/diedrichsenlab/multitaskbattery:latest

For a study, use a fixed release instead of ``latest`` (e.g.
``multitaskbattery:1.2.0``), so every session and site runs exactly the same
version.

``docker/run_docker.sh`` in the repository starts the container with access to
the screen, sound and the scanner keyboard, and runs a Python script in it::

    bash docker/run_docker.sh <script.py>
