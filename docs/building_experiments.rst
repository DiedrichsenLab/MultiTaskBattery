Building new experiments
========================

To build a new experiment, first create a new project folder somewhere on your computer. Based on the example experiment ``expertiments/example_experiment``, follow the steps below:

Step 1: Constants file
--------------
Create a file called ``constants.py`` in the project folder. This file contains information pertaining to the scanner, screen, response device, and pointers to the local directories. If you run the experiment in multiple setups, it is useful to create a differnt versions of this file, for example `constants_frmi.py` and a `constants_behavioral.py`.

.. code-block:: python

    # constants.py defines parameters and settings for an experiment
    # it is passed to the Experiment class on initialization
    from pathlib import Path
    import os
    import MultiTaskBattery as mtb

    #Necessary definitions for the experiment:
    exp_name = 'excample_experiment'

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


Step 2: Generating run and task files
-----------------------------
Task and run files are tab-delimited text files (``.tsv``) that specify the order of task in each run, and the order of trials within each task.
Then create and run a small Python script to generate your run and task files. Very basic examples are included in the example_experiment. Depending on your experiment, you may want to add more information. Of course you can produce these files by hand, but we prefer to write a function in ``task_files.py`` that does the randomization for us.

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

* some of the tasks require run number because the stimuli depend on the run number (e.g., movie clips have a specific order for each run)

**Example Code**

.. code-block:: python

    tasks = ['n_back','rest'] # ,'social_prediction','verb_generation'
    for r in range(1,9):
        tfiles = [f'n_back_{r:02d}.tsv','rest_30s.tsv'] # f'social_prediction_{r:02d}.tsv',f'verb_generation_{r:02d}.tsv',
        T  = mt.make_run_file(tasks,tfiles)
        T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

        # for each of the runs, make a target file
        for task,tfile in zip(tasks, tfiles):
            cl = mt.get_task_class(task)
            myTask = getattr(mt,cl)(const)
            myTask.make_trial_file(file_name = tfile)


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
        # main(sys.argv[1])
        main('sub-01')
