Implementing new tasks
======================

If you would like to implement your own tasks in addition to the once we have provided here, you can follow the steps below.

Register the task in the task_table
-----------------------------------
Start by adding adding your task name,task code (unique code for each task), the class, description, and reference to the `task_table.tsv` file in the `MultiTaskBattery` directory. This keeps track of all the tasks in the battery.


Generate new task files
-----------------------
Add a new class to the task_file.py file in the `MultiTaskBattery` directory. This class should inherit from the `TaskFile` class and should have the following methods:

- `__init__`: This method should call the `__init__` method of the parent class and set the `self.task_name` attribute to the name of the task.
- `make_task_file`: This method should create a single task file for the task.

Implement the task code
-----------------------
Add a new class to the task_blocks.py module in the `MultiTaskBattery` directory. This class should inherit from the `Task` class and should have the following functions:

- `init_task`: Initializes the task. The default behaviour is to read the target information into the trial_info dataframe. Additionally, you may want to load any stimuli required.
- `display_instructions`: Displays the instruction for the task. Most tasks have the same instructions, giving information about what to do and which keys to use for responding. Those tasks that have different instructions will have their own display_instructions method.
- `run_trial`: This function loops over trials and collects data. This is where you define what will be displayed during the trial, which responses will be collected and how these responses are processed. The collected data will be stored in self.trial_data

In creating your task, make use of the existing task routines that are defined in the `Task` class:

- `wait_response`: Waits for a response to be made and then returns the response.
- `display_trial_feedback`: Displays the feedback for the current trial using the color of the fixation cross
- `save_data`: Saves the data to the trial data file.
- `screen_quit`: Checks for quit or escape key presses and quits the experiment if necessary.

