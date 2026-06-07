# constants.py defines parameters and settings for an experiment
# it is passed to the Experiment class on initialization
#
# This example shows how to extend MultiTaskBattery with tasks defined
# locally in your experiment folder (no need to edit the shared package).
# See my_tasks.py (holds both Task and TaskFile classes) and the local
# task_table.tsv.

from pathlib import Path
import os
import MultiTaskBattery as mtb
import my_tasks  # local module: both Task and TaskFile classes for custom tasks

#Necessary definitions for the experiment:
exp_name = 'example_custom_task'

#UNCOMMENT THIS FOR SCANNING
#response_keys    = ['y', 'g', 'r', 'm'] # scanner keys

#COMMENT THIS FOR SCANNING
response_keys    = ['a', 's', 'd', 'f']

response_fingers = ['Pinky', 'Ring','Middle', 'Index']

# Directory definitions for experiment
exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))
task_dir = exp_dir / "task_files"
run_dir    = exp_dir / "run_files"
data_dir   = exp_dir / "data"

# Modules where MultiTaskBattery will look for custom Task classes at
# runtime. Anything not found here falls back to MultiTaskBattery.task_blocks.
# TaskFile classes for custom tasks live in the same module(s), using a
# 'File' suffix on the class name.
task_modules = [my_tasks]

default_run_filename = 'run_{}.tsv'

package_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(mtb.__file__))))
stim_dir   = package_dir / "stimuli"

eye_tracker = False
debug = False

screen = {}
screen['size'] = [1100, 800]
screen['fullscr'] = False
screen['number'] = 1
