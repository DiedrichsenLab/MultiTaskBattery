# constants.py defines parameters and settings for an experiment
# it is passed to the Experiment class on initialization
from pathlib import Path
import os

#Necessary definitions for the experiment:
exp_name = 'pontine_7T'

# Response key assignment:
response_keys    = ['a', 's', 'd', 'f']
response_fingers = ['Pinky', 'Ring','Middle', 'Index']

# Directory definitions for experiment
exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))   # where the experiment code is stored
target_dir = exp_dir / "target_files"  # contains target files for the task
run_dir    = exp_dir / "run_files"     # contains run files for each session
data_dir   = exp_dir / "data"          # This is where the result files are being saved
stim_dir   = exp_dir /'..'/ "stimuli"       # This is where the stimuli are stored

# Eye tracker?
eye_tracker = False                                     # do you want to do the eyetracking?

# Running in debug mode?
debug = True                                           # set to True for debugging

# Screen settings for subject display
screen = {}
screen['size'] = [800, 400]                             # screen resolution
screen['fullscr'] = False                               # full screen?
screen['number'] = 1                                    # 0 = main display, 1 = secondary display
