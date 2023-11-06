Installation
============

Cloning this Repository
-----------------------

1. Copy the git repo URL. Click the "Clone or Download" button and copy the link (`https://github.com/diedrichsenlab/mdtb_reduced.git`).
2. Go to your terminal and navigate (using `cd` and `ls` commands) to the directory where you want to clone the repository.
3. Use `git clone` to download the entire folder to your computer:

```
git clone https://github.com/diedrichsenlab/mdtb_reduced.git
```

Or use Gitdesktop to clone the repository.

### Installing the Required Python Version

This project requires **python version >3.7.0**. Please ensure it is installed globally on your local machine.

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

Installing the Required Python Packages
---------------------------------------

It is a good idea to create a virtual environment, and install the dependencies in this environment. This can be done using `pipenv`, or to make it more convient virtualenvwrapper.

required packages:
pip install psychopy 
pip install psychopy-mri-emulator

Installs the required packages in the virtual environment with requirements. 
