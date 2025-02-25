Custom Experiments
========================

To build a new experiment, first create a new project folder somewhere on your computer. Based on the example experiment ``expertiments/example_experiment``, follow the steps below:

Constants file
--------------
Create a file called ``constants.py`` in the project folder. This file contains information pertaining to the scanner, screen, response device, and pointers to the local directories. If you run the experiment in multiple setups, it is useful to create a differnt versions of this file, for example `constants_frmi.py` and a `constants_behavioral.py`.

.. code-block:: python

    # constants.py defines parameters and settings for an experiment
    # it is passed to the Experiment class on initialization
    from pathlib import Path
    import os
    import MultiTaskBattery as mtb

    #Necessary definitions for the experiment:
    exp_name = 'example_experiment'

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


Generating run and task files
-----------------------------
Task and run files are tab-delimited text files (``.tsv``) that specify the order of task in each run, and the order of trials within each task.
Then create and run a small Python script to generate your run and task files. Very basic examples are included in example_experiment/make_files.py. Depending on your experiment, you may want to add more information. Of course you can produce these files by hand, but we prefer to write a function in ``task_files.py`` that does the randomization for us.

**Run Files**
Run files that specify the structure of the runs, including the order of the tasks for the run, which task file contains the stimuli for this run.

Each run file should contain the following columns:
- task_name: Name of the task
- task_code: short name of the task
- task_file: Name of the task file for this run
- instruction_dur: Duration of the instruction screen before the task starts (in seconds)
- start_time: Start time of the task (in seconds from the start of the run)
- end_time: End time of the task (in seconds)

**Task Files**
Task files that specify the structure of the tasks within each run (e.g. the stimuli, the correct response, whether to display feedback, etc.).

The task file can look very different form tasks to task, but typically contains some of the following columns:

- trial_num: Trial number
- hand: Hand used for the task (left or right)
- trial_dur: Duration of the trial (in seconds)
- iti_dur: Inter-trial interval duration (in seconds)
- stim: Stimulus presented
- display_trial_feedback: Whether to display feedback after each trial
- start_time: Start time of the trial (in seconds)
- end_time: End time of the trial (in seconds)
- Key columns, for example in the case of four response keys (e.g. in the RMET task or Finger Sequence task):
  - key_one: Key for the first option
  - key_two: Key for the second option
  - key_three: Key for the third option
  - key_four: Key for the fourth option

Some of the tasks require run number because the stimuli depend on the run number (e.g., movie clips have a specific order for each run)

**Example Code**

.. code-block:: python

    import MultiTaskBattery.task_file as tf
    import MultiTaskBattery.utils as ut
    import constants as const

    tasks = ['finger_sequence', 'n_back', 'demand_grid', 'auditory_narrative',
         'sentence_reading', 'verb_generation', 'action_observation',
         'tongue_movement', 'theory_of_mind', 'rest']

    num_runs = 8  # Number of imaging runs

    # Ensure task and run directories exist
    ut.dircheck(const.run_dir)
    for task in tasks:
        ut.dircheck(const.task_dir / task)

    # Generate run and task files
    for r in range(1, 11):
        tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
        T = tf.make_run_file(tasks, tfiles)
        T.to_csv(const.run_dir / f'run_{r:02d}.tsv', sep='\t', index=False)

        # Generate a target file for each run
        for task, tfile in zip(tasks, tfiles):
            cl = tf.get_task_class(task)
            myTask = getattr(tf, cl)(const)

            # Add run number if necessary
            args = {}
            if myTask.name not in ut.tasks_without_run_number:
                args.update({'run_number': r})

            # Make task file
            myTask.make_task_file(file_name=tfile, **args)
         
> Note that you can add an optional argument run_time to the make_task_file function to specify the duration of your run (e.g. ``myTask.make_task_file(tasks, tfiles, run_time=600)`` for a 10-minute run). After the last trial ends, this will return the screen to a fixation cross until the run_time is reached. This is usfeul for imaging experiments where you want to keep the scanner running for a fixed amount of time after the last trial to capture the remaining activation. If this is not specified, the run will end after the last trial.
> You can also add an optional argument offset to the make_task_file function to start the stimuli presentation after some seconds of fixation cross  (e.g. ``myTask.make_task_file(tasks, tfiles, offset=5)`` for a 5-second delay after the first trigger). This is recommended for imaging experiments where you acquire dummy scans in the beginning of the scan (to account for stabilizing magnetization) that will be removed from the data in later processing. If during those dummy scans trigger signals are already being sent out, this will have the first stimulus presented only after this offset period accounting for dummy scans has passed. If the offset parameter has not been specified, the run will end after the last trial.

Writing your experiment function
--------------------------------

After generating the tasks and run files, you can write your own main script to run the experiment.

.. code-block:: python

    import sys
    import MultiTaskBattery.experiment_block as exp_block
    import constants as const

    def main(subj_id):
        """_summary_
        make sure you to adjust constanst.py file before running the experiment
        (e.g., experiment_name, eye_tracker, screen, etc.)

        Args:
            subj_id (str): id of the subject
        """
        my_Exp = exp_block.Experiment(const, subj_id=subj_id)

        while True:
            my_Exp.confirm_run_info()
            my_Exp.init_run()
            my_Exp.run()
        return

    if __name__ == "__main__":
        main()
