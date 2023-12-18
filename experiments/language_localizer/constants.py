# created 2023: Bassel Arafat, Jorn Diedrichsen
# constants.py defines parameters and settings for an experiment
# it is passed to the Experiment class on initialization

from pathlib import Path
import os
import MultiTaskBattery as mtb

#Necessary definitions for the experiment:
exp_name = 'language_localizer'

# Response key assignment:
response_keys    = ['a', 's', 'd', 'f']
response_fingers = ['Pinky', 'Ring','Middle', 'Index']

# Directory definitions for experiment
exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))   # where the experiment code is stored
task_dir = exp_dir / "task_files"  # contains target files for the task
run_dir    = exp_dir / "run_files"     # contains run files for each session
data_dir   = exp_dir / "data"          # This is where the result files are being saved

package_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(mtb.__file__))))
stim_dir   = package_dir / "stimuli"       # This is where the stimuli are stored

# is the Eye tracker being used
eye_tracker = False                                     # do you want to do  eyetracking?

# Running in debug mode?
debug = False                                           # set to True for debugging

# Making real files or mock training files?
training = False                                        # set to True for training

# Screen settings for subject display
screen = {}
screen['size'] = [1100, 800]                             # screen resolution
screen['fullscr'] = False                               # full screen?
screen['number'] = 1                                    # 0 = main display, 1 = secondary display

