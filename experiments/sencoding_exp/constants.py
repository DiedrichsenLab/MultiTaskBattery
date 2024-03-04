# created 2023: Bassel Arafat, Jorn Diedrichsen
# constants.py defines parameters and settings for an experiment
# it is passed to the Experiment class on initialization

from pathlib import Path
import os
import MultiTaskBattery as mtb

#Necessary definitions for the experiment:
exp_name = 'sencoding_exp'

# Response key assignment:
# response_keys    = ['y', 'g', 'r', 'm'] # scanner keys
response_keys    = ['a', 's', 'k', 'l'] # behavioral keys keys
response_fingers = ['Pinky', 'Ring','Middle', 'Index']

# Directory definitions for experiment
exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))   # where the experiment code is stored
task_dir = exp_dir / "task_files"  # contains target files for the task
run_dir    = exp_dir / "run_files"     # contains run files for each session
data_dir   = exp_dir / "data"          # This is where the result files are being saved

# do run_file_name as a formated string
default_run_filename = 'sub-06_ses-01_run_{}.tsv'

package_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(mtb.__file__))))
stim_dir   = package_dir / "stimuli"       # This is where the stimuli are stored

# is the Eye tracker being used
eye_tracker = False                                     # do you want to do  eyetracking?

# Running in debug mode?
debug = False                                           # set to True for debugging

# Screen settings for subject display
screen = {}
screen['size'] = [1024, 768]                             # screen resolution
screen['fullscr'] = False                               # full screen?
screen['number'] = 1                                    # 0 = main display, 1 = secondary display

