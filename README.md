mdtb_reduced
==============================

This project uses a reduced battery of cognitive and motor tasks to map
the functional sub-regions of the human cerebellum

The tasks are Visual Search, Action Observation, N Back,
Social Prediction, and Semantic Prediction

#### Authors: Maedbh King

## Installation

### Cloning this Repository

1. Copy the git repo URL. Click the "Clone or Download" button and copy the link (`https://github.com/maedbhk/mdtb_reduced.git`).
2. Go to your terminal and navigate (using `cd` and `ls` commands) to the directory where you want to clone the repository. 
3. Use `git clone` to download the entire folder to your computer:
```
git clone https://github.com/maedbhk/mdtb_reduced.git
```

> NOTE: If you want to learn more about git, check out this tutorial [here](https://rogerdudler.github.io/git-guide/).

### Installing the Required Python Version

This project requires **python version 3.7.0**. Please ensure it is installed globally on your local machine.

If you are running Mac OS X or Linux, it is recommended to use [`pyenv`](https://github.com/pyenv/pyenv)
for python version management. The full installation instructions can be found [here](https://github.com/pyenv/pyenv#installation). 

Below is an abridged version of the `pyenv` install procedure for Max OS X:

Install `pyenv` using Homebrew:

    $ brew update
    $ brew install pyenv

Add `pyenv init` to your shell:

    $ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
    $ source ~/.bash_profile

Install the required version of python:

    $ pyenv install 3.7.0

### Installing the Required Python Packages

This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management.

Ensure pipenv is installed globally:

    $ brew install pipenv

Navigate to the top-level directory in `mdtb_reduced` and install the packages from the `Pipfile.lock`.
This will automatically create a new virtual environment for you and install all requirements using the correct version of python.

    $ pipenv install

## Running an Experiment

First, activate the virtual environment:

    $ pipenv shell

> NOTE: To deactivate the virtual environment when you are done working, simply type `exit`

Next, retrieve stimulus files:

    $ Download the folders: stimuli, target_files and run_files from the shared folder on google drive and save in top-level directory of the mdtb_reduced folder
    $ Eventually these folders will be available on AWS

To start a new experiment, execute:

    $ run-fmri or run-behavioral
    
After running the above command, a GUI will open with the following inputs: 

1. subject_id: example: `s01` 
2. study_name: example: `fmri` or `behavioral`
3. run_name:   example: `run_01`

To generate target and run files, execute:

    $ makefiles-fmri or makefiles-behavioral

All parameters are set in the __init__ methods of the target and run cases in `make_target_run.py`

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been preprocessed.
    │   ├── processed      <- The final, canonical data used for analysis
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── Pipfile            <- The Pipfile for reproducing the analysis environment, e.g.
    │                         generated with `pipenv install` or `pipenv --python 3.7.0`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── experiment_code    <- experiment code for use in this project.
    │   ├── __init__.py    <- Makes experiment_code a Python module
    │   │
    │   ├──                <- Scripts to run both behavioral and fmri experiments and make target and run files
    │   │   │── constants.py
    │   │   │── task_blocks
    │   │   │── screen.py       
    │   │   ├── make_target_run.py
    │   │   └── run_experiment.py
    │   │   ├── ttl.py
    │   │
    │   │
    │   ├── scripts         <- Scripts to run the experiments
    │   │   │                 
    │   │   ├── run-fmri.py
    │   │   └── run-behavioral.py
    │   │
    │   └── visualization  <- Scripts to quickly analyze behavioral data from experiments             
    │      ├── behavioral_visualize.py
    │      └── fmri_visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
