mdtb_reduced
==============================

This project uses a reduced battery of cognitive and motor tasks to map
the functional sub-regions of the human cerebellum

The tasks are Visual Search, Action Observation, N Back,
Social Prediction, and Semantic Prediction

#### Authors: Maedbh King, Ladan Shahshahani

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

    $ Download the folders: stimuli from the server
    
### Installing psychopy
Alternatively, you can follow the isntructions on https://www.psychopy.org/download.html#conda to create a virtual environment for psychopy. If you choose to do so, each time before running the experiment, you need to use conda to activate the psychopy virtual environment by typing:

    $ conda activate psychopy
    
## before you start:
1. go to experiment_code/constants and change experiment_name and base_dir. For the pontine project, experiment_name = 'pontine_7T' and base_dir = Path('where my base directory is').absolute() 
2. make sure 'stimuli' folder is located under your base_dir
3. run constants.dirtree() to make sure you have all the folders

## coding your experiment
use pontine_7T.py as an example and build the code for your experiment.
### debugging the code
Take pontine_7T.py as an example. run routine in this module has an input called 'debug'. The default value of this input is set to True. For debugging your code, make sure you debug is set to True. And for running the code during training and scan, make sure that debug is set to False. 

## running an experiment
Start a python prompt

    $ import python_7T.python_7T as e
    
if you haven't created target files

    $ e.create_target()
    
    * you can play around with the task_list variable. Choose tasks that are already defined in task_block.py
    * strings representing task names in the list should exist in the TASK_MAP variable in task_block.py
run the experiment code:
### for debugging:

    $ e.run(debug = True)
    
    * you will be prompted to enter the run number in the terminal. Enter the number for the run you want to test
    * you may change 'behav_training' and 'ttl_flag' as well.
### once you have debugged the code:

    $ e.run(debug = False)
    * a dialogue box will pop up asking you for experiment parameters
### to run or test the code for fmri, you can write a 'fmri_simulator.py' and do:

    $ import python_7T.fmri_simulator
    * a dialogue box will pop up asking for scanning parameters. This code will simulate ttl pulses and can be used for checking ttl syncing.
    * While running the code, always check the output printed in the terminal


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── <experiment_name>  <- Data from third party sources.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
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
    │   │   │── task_blocks.py
    │   │   │── screen.py       
    │   │   ├── make_target.py
    │   │   └── experiment_block.py
    │   │   ├── ttl.py
    │   │
    ├── <experiment_name>    <- a folder with the name you have chosen for the experiment. Example: pontine_7T
    │   │
    │   ├──                
    │   │   │── <expperiment_name>.py   <- Scripts to create files and run the experiment. Example: pontine_7T.py
    │   │   ├── fmri_simulator.py       <- Script to run/test the code for scanner
    │   │   ├── run_files               <- folder containing run files for your experiment
    │   │   ├── target_files            <- folder containing target files for the tasks in your experiment
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
