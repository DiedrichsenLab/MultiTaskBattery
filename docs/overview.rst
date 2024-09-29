Overview
========

Repository Structure
--------------------

The main code and classes, including the implementation of all the tasks, is in modules in the folder ``MultiTaskBattery``.  Stimuli are defined in the folder ``stimuli``, with a separate subfolder for each task. To develop and run your own experiment, we will create a new subfolder in the ``experiments`` folder. The ``pontine_7T`` experiment that is included in the main branch of the repository serves as an example of how to set up your own study.


The repository has the following Structure:

::

|-MultiTaskBattery: Main Python modules and classes
|-Stimuli: Stimuli used in the experiment
   |-n_back: Stimuli for n_back tasks
   |- ...
|-experiments: Folder for each experiment
   |- pontine_7T: Specific Experiment (example)
      |-pontine_7T.py: Your main Pyhton program
      |-constants.py: Constants for the experiment / scanner
      |-make_files.py: Script to pre-randomize task and run files
      |-run_files: Files specifying which tasks are done in which run (and which order)
      |-task_files: Files specifying which trials are done for each task block
      |-data: Data files for each subject

Program Structure
-----------------

.. image:: assets/flow_diagram.png
  :width: 700

Your main program will create and experiment object - with all settings specified in the constants.py file.

.. code-block:: python

    import constants as const
    my_Exp = exp_block.Experiment(const, subj_id=subj_id)

    while True:
        my_Exp.confirm_run_info()
        my_Exp.init_run()
        my_Exp.run()
    return

``confirm_run_info()`` will ask the user to confirm the run information.
``init_run()`` will read the run_file and then create the task objects, which in turn will read the trial files.

``Experiment.run()`` will run a run of the experiment, which then calls ``Task.run()``, which then calls ``Task.run_trial()``. For most tasks, only the latter function needs to be defined.

