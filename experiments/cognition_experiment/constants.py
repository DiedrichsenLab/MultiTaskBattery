from pathlib import Path
import os
import MultiTaskBattery as mtb

#Necessary definitions for the experiment:
exp_name = 'cognition_experiment'

# These are the response keys (change depending on your keyboard)
response_keys    = ['a', 's', 'd', 'f']

# Directory definitions for experiment
exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))   # where the experiment code is stored
task_dir = exp_dir / "task_files"  # contains target files for the task
run_dir    = exp_dir / "run_files"     # contains run files for each session
data_dir   = exp_dir / "data"          # This is where the result files are being saved

# This is were the stimuli for the different task are stored
package_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(mtb.__file__))))
stim_dir   = package_dir / "stimuli"

# do run_file_name as a formated string
default_run_filename = 'run_01.tsv'

# Is the Eye tracker being used?
eye_tracker = False

# Running in debug mode?
debug = False # set to True for debugging

# Screen settings for subject display
screen = {}
screen['size'] = [1100, 800]        # screen resolution
screen['fullscr'] = False           # full screen?
screen['number'] = 1                # 0 = main display, 1 = secondary display