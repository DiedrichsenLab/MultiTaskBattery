Overview
========

Repository Structure
--------------------

The main code and classes, including the implementation of all the tasks, is in modules in the folder ``MultiTaskBattery``.  Stimuli are defined in the folder ``stimuli``, with a separate subfolder for each task. To develop and run your own experiment, we will create a new subfolder in the ``experiments`` folder. The ``example_experiment`` folder gives an example of how to set up your own study.

::

|-MultiTaskBattery: Main Python modules and classes
|-Stimuli: Stimuli used in the experiment
|  |-n_back: Stimuli for n_back tasks
|     |- ...
|-experiments: Folder for each experiment
|  |- example_experiment: Specific Experiment (example)
|     |-example_experiment.py: Your main Python program
|     |-constants.py: Constants for the experiment setup / scanner
|     |-make_files.py: Script to pre-randomize task and run files
|     |-run_files: Files specifying which tasks are done in which run (and which order)
|     |-task_files: Files specifying which trials are done for each task block
|     |-data: Data files for each subject

Program Structure
-----------------

.. image:: assets/flow_diagram.png
  :width: 700

The experiment is controlled at at two levels. At the ``run_xx.tsv`` file specifies which tasks are used in each run, and which order and with what timing they are presented in. Typically we recommend to run each task for 30-40s in a randomized order. Each task then has a separate ``task_xx.tsv`` file that determines the exact order of trials within a task, as well as the stimuli and other details.

Your ``main`` program will create an ``Experiment`` object - with all settings specified in the ``constants.py`` file. Because these settings depend exact setup that is used, we usually have different ``constant.py`` for behavioral training and scanning,

.. code-block:: python

    import constants as const
    my_Exp = exp_block.Experiment(const, subj_id=subj_id)

    while True:
        my_Exp.confirm_run_info()
        my_Exp.init_run()
        my_Exp.run()
    return

``confirm_run_info()`` will ask the user to confirm the run information.

``init_run()`` will read the run_file and then create the task objects, which in turn will read the trial files. Run and task files are created before the experiment starts, using the specification in your ``make_files.py`` module.

``Experiment.run()`` will finally execute a run of the experiment, calling ``Task.run()`` for each task in the right moment, which then calls ``Task.run_trial()``. For most tasks, only the latter function needs to be defined. The data for each task will be collected and saved in the file ``<subj_id>_<task_id>.tsv`` in the data folder. The actual timing of the different tasks in each run are stored in the ``<subj_id>.tsv`` file. Both files collect the data from all the runs in an experiment, so it is important that you use unique run numbers.

Implemented tasks
-----------------

The task that are implemented in the repository are listed in the ``task_table.tsv`` file. This is also where new task have to be added and linked to a task class, so the program knows which class to instantiate when creating the task objects. Each class has an official name and a short code, which is used in the data files. For full description of the tasks, see `ref:tasks_instructions`.

.. csv-table:: List of tasks
   :file: task_table_for_docs.csv
   :widths: 30,30,120
   :header-rows: 1

