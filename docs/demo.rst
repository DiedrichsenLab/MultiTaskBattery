Demo Experiment
========================

For this demo, we will build an experiment that contains **10 tasks** targeting diverse cognitive domains. The experiment will include a total of **8 imaging runs**.


Step 1: Create the Experiment Folder
-------------------------------------
1. Create a new folder for your experiment. For this demo, we will call it `cognition_experiment`.
2. Add the folder to the `experiments` directory.
3. Add the following **empty files** to the folder:
   - **`constants.py`**: Contains information about the scanner, screen, response device, and pointers to the local directories.
   - **`make_files.py`**: A Python script to generate task and run files.
   - **`cognition_experiment.py`**: A Python script to run the experiment.

---

Step 2: Choose Tasks
---------------------
Use the task library available at:
   [MultiTaskBattery Documentation](https://multitaskbattery.readthedocs.io/en/latest/overview.html#implemented-tasks)

**Tasks for the Demo**:

- `finger_sequence`
- `n_back`
- `demand_grid`
- `auditory_narrative`
- `sentence_reading`
- `verb_generation`
- `action_observation`
- `tongue_movement`
- `theory_of_mind`
- `rest`

---

Step 3: Fill in the `constants.py` File
---------------------------------------
Fill in the `constants.py` file with the following content and adjust as needed:

.. code-block:: python

    from pathlib import Path
    import os
    import MultiTaskBattery as mtb

    # Necessary definitions for the experiment:
    exp_name = 'cognition_experiment'

    # Response keys (change depending on your keyboard)
    response_keys = ['a', 's', 'd', 'f']

    # Directory definitions for the experiment
    exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))   # Experiment code location
    task_dir = exp_dir / "task_files"  # Task files directory
    run_dir = exp_dir / "run_files"    # Run files directory
    data_dir = exp_dir / "data"        # Output data directory

    # Stimuli directory (stored in the package directory)
    package_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(mtb.__file__))))
    stim_dir = package_dir / "stimuli"

    # Default run file name
    default_run_filename = 'run_01.tsv'

    # Eye tracker usage
    eye_tracker = False

    # Debug mode
    debug = False  # Set to True for debugging

    # Screen settings for subject display
    screen = {
        'size': [1100, 800],       # Screen resolution
        'fullscr': False,          # Full screen mode
        'number': 1                # 0 = main display, 1 = secondary display
    }

---

Step 4: Generate Run and Task Files
-----------------------------------
Fill in the `make_files.py` file. Here we will generate **run** and **task** files for the 8 runs and 10 tasks.

You can also add conditional rules to manage task order. For example, to avoid a motor control task (like `finger_sequence`) being immediately followed by another motor control task (like `tongue_movement`).

**Example Code**:

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

---

Step 5: Write the Experiment Function
-------------------------------------
After generating the task and run files, fill in the `cognition_experiment.py` file with the following code:

.. code-block:: python

    import sys
    import MultiTaskBattery.experiment_block as exp_block
    import constants as const

    def main(subj_id):
        """ Main experiment function.
        Ensure the constants.py file is updated before running the experiment
        (e.g., experiment name, eye tracker, screen settings, etc.).

        Args:
            subj_id (str): Subject ID
        """
        my_Exp = exp_block.Experiment(const, subj_id=subj_id)

        while True:
            my_Exp.confirm_run_info()
            my_Exp.init_run()
            my_Exp.run()
        return

    if __name__ == "__main__":
        main('subject-0')

---

Step 6: Run the Experiment
---------------------------
Specify the **subject ID** and execute the script. Output files will be saved in the `data` folder using the subject ID as part of the filename.

**Example**:

.. code-block:: bash

    python cognition_experiment.py

**Output**: Data will be saved in `data/` directory with the subject ID as the filename prefix.

---