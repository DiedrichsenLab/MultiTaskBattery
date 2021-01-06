# import libraries
from pathlib import Path
import os

response_keys = ['d', 'f', 'j', 'k']

# assign keys to hands
key_hand_dict = {
    'right': {    # right hand
        True:  [response_keys[2], 'Index'], # index finger
        False: [response_keys[3], 'Middle'],  # middle finger
        },
    'left': {   # left hand
        False:[response_keys[0], 'Middle'], # index finger
        True: [response_keys[1], 'Index'],  # middle finger
        },
    } 

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