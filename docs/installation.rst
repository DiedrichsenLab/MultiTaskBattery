Installation
============

Step 1: Clone the Repository
----------------------------

Using Git::

    git clone https://github.com/diedrichsenlab/MultiTaskBattery.git
    cd MultiTaskBattery

Or use `GitHub Desktop <https://desktop.github.com/>`_.

Step 2: Install Python (3.10)
-----------------------------

This project uses **Python 3.10**.

Option A: Using pyenv (Recommended on macOS/Linux)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install pyenv::

    brew update
    brew install pyenv

Configure your shell::

    echo 'if command -v pyenv 1>/dev/null 2>&1; then eval "$(pyenv init -)"; fi' >> ~/.bash_profile
    source ~/.bash_profile

Install Python::

    pyenv install 3.10
    pyenv global 3.10

Option B: Using system Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure ``python --version`` reports 3.10.

Step 3: Create a Virtual Environment
------------------------------------

::

    python -m venv mtb-env
    source mtb-env/bin/activate    # On Windows: mtb-env\Scripts\activate

Step 4: Install Dependencies
----------------------------
Make sure you upgrade pip first, then install the required packages.

::

    pip install --upgrade pip
    python install.py

``install.py`` installs everything from ``requirements.txt``. On Linux it also
detects your distribution and fetches a prebuilt wxPython wheel (wxPython has no
Linux wheels on PyPI, so a plain pip install would fail trying to compile it).
On Windows and macOS it is equivalent to ``pip install -r requirements.txt``,
which you can still use directly.

Step 5: Add to Python Path
--------------------------
Add the repository to your Python path so that ``MultiTaskBattery`` can be imported from anywhere. Add the following line to your shell profile (e.g., ``~/.bashrc``, ``~/.zshrc``, or ``~/.bash_profile``)::

    export PYTHONPATH="/path/to/MultiTaskBattery:$PYTHONPATH"

On Windows, add the repository path to the ``PYTHONPATH`` environment variable via System Settings > Environment Variables.

Replace ``/path/to/MultiTaskBattery`` with the actual path to the cloned repository.

