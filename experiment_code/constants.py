# import libraries
from pathlib import Path
import os

class Defaults: 

    def __init__(self):
        # set response keys
        self.RESPONSE_KEYS = ['d', 'f', 'j', 'k']

        # assign keys to hands
        self.KEY_HAND_DICT = {
        'right': {    # right hand
            True:  [self.RESPONSE_KEYS[2], 'Index'], # index finger
            False: [self.RESPONSE_KEYS[3], 'Middle'],  # middle finger
        },
        'left': {   # left hand
            False:[self.RESPONSE_KEYS[0], 'Middle'], # index finger
            True: [self.RESPONSE_KEYS[1], 'Index'],  # middle finger
        },
        } 
class Directories:

    def __init__(self):
        self.BASE_DIR = Path(__file__).absolute().parent.parent
        self.STIM_DIR   = self.BASE_DIR / "stimuli"
        self.TARGET_DIR = self.BASE_DIR / "target_files"
        self.RUN_DIR    = self.BASE_DIR / "run_files"
        self.RAW_DIR    = self.BASE_DIR / "data"

    def _dircheck(self, path2dir):
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

    def _dirtree(self):
        """
        Create all the directories if they don't already exist
        """
        fpaths = [self.RAW_DIR, self.STIM_DIR, self.TARGET_DIR, self.RUN_DIR]
        for fpath in fpaths:
            self._dircheck(fpath)