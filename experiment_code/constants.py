# import libraries
from pathlib import Path
import os

# default response keys and the corresponding fingers. Can be modified
response_keys    = ['a', 's', 'd', 'f', 'h', 'j', 'k', 'l']
response_fingers = ['Pinky', 'Ring', 'Middle', 'Index', 'Index', 'Middle', 'Ring', 'Pinky']

base_dir   = Path(__file__).absolute().parent.parent
stim_dir   = base_dir / "stimuli"
target_dir = base_dir / "target_files"
run_dir    = base_dir / "run_files"

# This is where the result files are being saved
raw_dir    = base_dir/ "data"  

def dircheck(path2dir):
    """
    Checks if a directory exists! if it does not exist, it creates it
    Args:
    dir_path    -   path to the directory you want to be created!
    """

    if not os.path.exists(path2dir):
        print(f"creating {path2dir}")
        os.makedirs(path2dir)
    else:
        print(f"{path2dir} already exists")

def dirtree():
    """
    Create all the directories if they don't already exist
    """
    fpaths = [raw_dir, stim_dir, target_dir, run_dir]
    for fpath in fpaths:
        dircheck(fpath)