# constants.py defines parameters and settings for an experiment
# it is passed to the Experiment class on initialization
from pathlib import Path
import os
import MultiTaskBattery as mtb
import task_olive as to

#Necessary definitions for the experiment:
exp_name = 'olive_7T' # name of the experiment

#UNCOMMENT THIS FOR SCANNING
#response_keys    = ['y', 'g', 'r', 'm'] # scanner keys

#COMMENT THIS FOR SCANNING
response_keys    = ['a', 's', 'd', 'f']

#not used
response_fingers = ['Pinky', 'Ring','Middle', 'Index']

# Directory definitions for experiment
exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))   # where the specific experiment code is stored
task_dir = exp_dir / "task_files"  # contains target files for the task
run_dir    = exp_dir / "run_files"     # contains run files for each session
data_dir   = exp_dir / "data"          # This is where the result files are being saved

# this is a list of imported task modules, which are used to search for the right task class 
task_modules = [to]

# Use {} so the GUI auto-fills the run number (e.g. run_01.tsv, run_02.tsv, ...)
default_run_filename = 'run_{}.tsv'

# This is were the stimuli for the different task are stored
package_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(mtb.__file__))))
stim_dir   = package_dir / "stimuli"

# Is the Eye tracker being used?
eye_tracker = False                                     # do you want to do  eyetracking?

# Running in debug mode?
debug = False                                           # set to True for debugging

# Responding hand: 'right' uses 'a'=yes/'s'=no; 'left' uses 'f'=yes/'d'=no
responding_hand = 'left'

# Screen settings for subject display
screen = {}
screen['size'] = [1024, 768]        # screen resolution
screen['fullscr'] = True           # full screen, if false it's in a separate window
screen['number'] = 1               # 0 = main display, 1 = secondary display
