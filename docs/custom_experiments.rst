Custom Experiments
========================

To build a new experiment, create a new folder for it **outside** the repository (not inside ``experiments/``), so that updates to MultiTaskBattery never interfere with your work. The stimuli directory is resolved automatically from the package location, so your experiment will find the stimuli regardless of where it lives. Use ``experiments/example_minimal`` as a reference when you only need built-in tasks, and ``experiments/example_custom_task`` when you also want to define your own tasks locally.

Constants file
--------------
Create a file called ``constants.py`` in the project folder. This file contains information pertaining to the scanner, screen, response device, and pointers to the local directories. If you run the experiment in multiple setups, it is useful to create different versions of this file, for example ``constants_fmri.py`` and ``constants_behavioral.py``.

.. code-block:: python

    # constants.py defines parameters and settings for an experiment
    # it is passed to the Experiment class on initialization
    from pathlib import Path
    import os
    import MultiTaskBattery as mtb

    #Necessary definitions for the experiment:
    exp_name = 'example_minimal'

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

    # Optional: list of local Python modules that hold custom Task / TaskFile
    # classes. Uncomment and import your module if you want to add custom
    # tasks without editing the shared MultiTaskBattery package. See
    # the "Implementing new tasks" page for details and
    # experiments/example_custom_task for a working example.
    # import my_tasks
    # task_modules = [my_tasks]

    # Use {} so the GUI auto-fills the run number (e.g. run_01.tsv, run_02.tsv, ...)
    default_run_filename = 'run_{}.tsv'

    # Is the Eye tracker being used?
    eye_tracker = False

    # Running in debug mode?
    debug = False # set to True for debugging

    # Screen settings for subject display
    screen = {}
    screen['size'] = [1100, 800]        # screen resolution
    screen['fullscr'] = False           # full screen?
    screen['number'] = 1                # 0 = main display, 1 = secondary display

Optional constants
^^^^^^^^^^^^^^^^^^

The following attributes can be added to ``constants.py`` to customise
experiment-wide behaviour.  They are all optional. If absent, sensible
defaults are used.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Attribute
     - Default
     - Description
   * - ``continue_key``
     - ``None``
     - If set to a key name (e.g. ``'space'``), only that key will dismiss
       the run-feedback scoreboard screen. If ``None``, any key continues.
   * - ``scoreboard_text_height``
     - ``1.3``
     - Height (in degrees of visual angle) of the text on the run-feedback
       scoreboard.  Reduce for smaller screens.
   * - ``record_run_end_timestamp``
     - ``False``
     - If ``True``, adds an ISO-format ``run_end_timestamp`` column to the run
       summary when each run completes (captured when the last task ends).
       Useful for post-processing that needs actual run-end wall-clock time.
   * - ``task_modules``
     - ``[]``
     - List of imported Python modules that hold custom Task and TaskFile
       classes for this experiment. ``ut.get_task_class`` and
       ``ut.get_task_file_class`` search these modules first, then fall back
       to the shared ``MultiTaskBattery`` package. See :doc:`creating_tasks`.
   * - ``instruction_text_height``
     - ``1``
     - Height (in degrees of visual angle) of the instruction-screen text
       shown before each task.  Reduce for smaller screens.

.. note::

   Display parameters are **per-task** and written into the task TSV files via
   ``make_task_file()`` parameters — including the on-screen size of video and
   image stimuli (``media_scale`` for video tasks, ``picture_scale`` for image
   tasks).  See the :ref:`task descriptions <task_descriptions>` page.  The only
   **experiment-wide** display settings that live in ``constants.py`` are
   ``instruction_text_height`` and ``scoreboard_text_height``.


Generating run and task files
-----------------------------
Task and run files are tab-delimited text files (``.tsv``) that specify the order of task in each run, and the order of trials within each task.
Create and run a small Python script to generate your run and task files. Basic examples are included in ``example_minimal/make_files.py`` and ``example_custom_task/make_files.py``. Depending on your experiment, you may want to add more information. Of course you can produce these files by hand, but we prefer to write a function that does the randomization for us.

For the columns that run files and task files contain, see the
:ref:`Run file columns <run file columns>` and
:ref:`Task file columns <task file columns>` references on the Getting
Started page. The general (shared) columns are described there; columns
specific to a single task are listed per task on the
:ref:`task descriptions <task_descriptions>` page.

Some tasks require a ``run_number`` because the stimuli depend on the run (e.g., movie clips have a specific order for each run). Tasks that generate random stimuli each run do not need a run number. The framework detects which case applies by inspecting the signature of each task's ``make_task_file`` — if it declares a ``run_number`` parameter, one is passed; otherwise it is not. To opt out, simply omit ``run_number`` from your task's ``make_task_file`` signature.

Each task's ``make_task_file`` accepts parameters that control the trial structure (e.g., grid size, trial duration, number of steps). See the :doc:`Task_file module reference <reference_task_file>` for each task's parameters and their defaults (the resulting task-file columns are documented on the :ref:`task descriptions <task_descriptions>` page). You can pass any of these as keyword arguments:

.. code-block:: python

    myTask.make_task_file(file_name=tfile, trial_dur=10, grid_size=(4, 5), **args)

Some tasks also have multiple **conditions** (e.g., ``movie`` has ``romance``, ``nature``, ``landscape``). If you want a specific condition, pass it as an argument to ``make_task_file``:

.. code-block:: python

    myTask.make_task_file(file_name=tfile, condition='romance', **args)

Check the :ref:`task descriptions <task_descriptions>` page to see which tasks have conditions.

**Example Code**

.. code-block:: python

    import inspect
    import MultiTaskBattery.task_file as tf
    import MultiTaskBattery.utils as ut
    import constants as const

    # (task, condition) pairs. Use None for single-condition tasks; pass an
    # explicit condition for tasks that have several (e.g. verb_generation,
    # reading) - each condition is generated as its own block.
    blocks = [('finger_sequence', None),
              ('n_back', None),
              ('demand_grid', None),
              ('auditory_narrative', None),
              ('reading', 'sentences'),
              ('verb_generation', 'read'),
              ('verb_generation', 'generate'),
              ('action_observation', None),
              ('tongue_movement', None),
              ('theory_of_mind', None),
              ('rest', None)]

    num_runs = 8  # Number of imaging runs

    # Ensure task and run directories exist
    ut.dircheck(const.run_dir)
    for task, cond in blocks:
        ut.dircheck(const.task_dir / task)

    # Generate run and task files
    for r in range(1, num_runs + 1):
        tasks = [task for task, cond in blocks]
        tfiles = [f'{task}_{cond}_{r:02d}.tsv' if cond else f'{task}_{r:02d}.tsv'
                  for task, cond in blocks]

        # Pass exp_dir so any local task_table.tsv (for custom tasks) is merged
        # with the framework's table.
        T = tf.make_run_file(tasks, tfiles, exp_dir=const.exp_dir)
        T.to_csv(const.run_dir / f'run_{r:02d}.tsv', sep='\t', index=False)

        # Generate task files for each run
        for (task, cond), tfile in zip(blocks, tfiles):
            cl = tf.get_task_class(task, exp_dir=const.exp_dir)

            # Looks up the TaskFile class: checks const.task_modules first
            # (for custom tasks), then falls back to MultiTaskBattery.task_file.
            TaskFileCls = ut.get_task_file_class(const, cl)
            myTask = TaskFileCls(const)

            # Pass condition only when the block has one, and run_number only if
            # the task's make_task_file accepts it.
            args = {}
            if cond is not None:
                args['condition'] = cond
            if 'run_number' in inspect.signature(myTask.make_task_file).parameters:
                args['run_number'] = r

            myTask.make_task_file(file_name=tfile, **args)

* Note that you can add an optional argument ``run_time`` to ``make_run_file`` to specify the duration of your run (e.g. ``tf.make_run_file(tasks, tfiles, run_time=600)`` for a 10-minute run). After the last trial ends, this will return the screen to a fixation cross until the run_time is reached. This is useful for imaging experiments where you want to keep the scanner running for a fixed amount of time after the last trial to capture the remaining activation. If this is not specified, the run will end after the last trial.
* You can also add an optional argument ``offset`` to ``make_run_file`` to start the stimuli presentation after some seconds of fixation cross (e.g. ``tf.make_run_file(tasks, tfiles, offset=5)`` for a 5-second delay after the first trigger). This is recommended for imaging experiments where you acquire dummy scans in the beginning of the scan (to account for stabilizing magnetization) that will be removed from the data in later processing. If during those dummy scans trigger signals are already being sent out, this will have the first stimulus presented only after this offset period accounting for dummy scans has passed.

Adding custom tasks to your experiment
--------------------------------------

You can add tasks that only exist for your experiment without modifying the shared ``MultiTaskBattery`` package.

1. Create a local module (e.g. ``my_tasks.py``) in your experiment folder that defines both a runtime ``Task`` subclass (e.g. ``MyTask``) and a matching ``TaskFile`` subclass with the ``File`` suffix (e.g. ``MyTaskFile``).
2. Add a row for the task to a local ``task_table.tsv`` in the same folder (same columns as the framework's table: ``name``, ``task_class``, ``descriptive_name``, ``code``).
3. In ``constants.py``, import the module and add it to ``task_modules``:

.. code-block:: python

    import my_tasks
    task_modules = [my_tasks]

At runtime, ``ut.get_task_class`` will find ``MyTask`` in your local module before checking the framework. At file-generation time, ``ut.get_task_file_class`` does the same and applies the ``File`` suffix automatically, so ``make_files.py`` does not need to change. 

See ``experiments/example_custom_task`` for a complete working example, and :doc:`creating_tasks` for the full structure of a task class.

Writing your experiment function
--------------------------------

After generating the tasks and run files, you can write your own main script `run.py` and save it in the project folder. This script will initialize the experiment and run it for a specific subject. Below is a basic example of how to structure this script:

.. code-block:: python

    import sys
    import MultiTaskBattery.experiment_block as exp_block
    import constants as const

    def main(subj_id):
        """_summary_
        make sure you to adjust constants.py file before running the experiment
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
        main('subject-00')
