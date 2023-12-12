# import libraries
from pathlib import Path
import os

# default response keys and the corresponding fingers. Can be modified
## first enter the keys for the right hand and then the keys for the left hand
# response_keys    = ['a', 's', 'd', 'f', 'h', 'j', 'k', 'l']
response_keys    = ['2', '3', '4', '5', '2', '3', '4', '5']
# response_keys    = ['y', 'g', 'r', 'e', 'y', 'g', 'r', 'e']
# ygrt
response_fingers = ['Index', 'Middle', 'Ring', 'Pinky', 'Index', 'Middle', 'Ring', 'Pinky']
# change experiment name to a name you've chosen for your own experiment
experiment_name = 'pontine_7T'
# change the str inside Path() to a directory of your choice.
## make sure 'stimuli' and 'experiment_code' folders are placed within your base_dir
base_dir        = Path('C:\\Users\\lshah\\OneDrive\\Documents\\Projects\\mdtb_reduced').absolute()

stim_dir   = base_dir / "stimuli"                       # where stimuli for each task are stored
target_dir = base_dir / experiment_name /"target_files" # contains target files for the task
run_dir    = base_dir / experiment_name /"run_files"    # contains run files for each session
raw_dir    = base_dir/ experiment_name / "data"         # This is where the result files are being saved

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

# use dirtree to make sure you have all the folders needed
def dirtree():
    """
    Create all the directories if they don't already exist
    """

    fpaths = [raw_dir, stim_dir, target_dir, run_dir]
    for fpath in fpaths:
        dircheck(fpath)