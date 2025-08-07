Installation
============

Step 1: Clone the Repository
----------------------------

Using Git::

    git clone https://github.com/diedrichsenlab/MultiTaskBattery.git
    cd MultiTaskBattery

Or use `GitHub Desktop <https://desktop.github.com/>`_.

Step 2: Install Python (≥ 3.9)
------------------------------

This project requires **Python 3.9 or later**.

Option A: Using pyenv (Recommended on macOS/Linux)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install pyenv::

    brew update
    brew install pyenv

Configure your shell::

    echo 'if command -v pyenv 1>/dev/null 2>&1; then eval "$(pyenv init -)"; fi' >> ~/.bash_profile
    source ~/.bash_profile

Install Python::

    pyenv install 3.9.0
    pyenv global 3.9.0

Option B: Using system Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure `python --version` reports 3.9 or higher.

Step 3: Create a Virtual Environment
------------------------------------

::

    python -m venv mtb-env
    source mtb-env/bin/activate    # On Windows: mtb-env\Scripts\activate

Step 4: Install Dependencies
----------------------------
Make sure you upgrade pip first, then install the required packages from the `requirements.txt` file.

::

    pip install --upgrade pip
    pip install -r requirements.txt

