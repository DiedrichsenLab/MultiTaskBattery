Building new experiments
========================

To build a new experiment, first create a new project folder somewhere on your computer. Based on the example experiment ``expertiments/pontine_7T``, follow the steps below: 

Constants file
--------------
Create a file called ``constants.py`` in the project folder. This file should contain the following information:

.. code-block:: python

    # constants.py defines parameters and settings for an experiment
    # it is passed to the Experiment class on initialization
    from pathlib import Path
    import os
    import MultiTaskBattery as mtb

    #Necessary definitions for the experiment:
    exp_name = 'pontine_7T'

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
Then create and run a small Python script to generate your run and task files. Here you specify which tasks and conditions you want to run. A very basic example is shown below - depending on your experiment, you may need to add more complexity. Before you run your experiment, check your files - they are simple human readable text files.

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

Providing a main function
-------------------------

The last step is to insert a small Python program that runs your experiment, which looks like this. 

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