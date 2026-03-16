Implementing new tasks
======================

Follow these steps to add a new task to the battery.

1. Register the task
--------------------
Add a row to ``MultiTaskBattery/task_table.tsv`` with:

- ``name``: task name using snake_case (e.g. ``serial_reaction_time``)
- ``task_class``: Python class name (e.g. ``SerialReactionTime``)
- ``descriptive_name``: short label for the GUI
- ``code``: unique short code (e.g. ``srt``)
- ``description``: brief description of the task
- ``reference``: academic citation (or ``NA``)
- ``conditions``: comma-separated conditions (or ``NA``)
- ``recorded_metrics``: what the task records — ``Accuracy + RT``, ``Accuracy``, ``RT``, or ``None``

2. Implement the task class
---------------------------
Add a new class to ``MultiTaskBattery/task_blocks.py`` that inherits from ``Task``. You need to implement:

- ``init_task()``: Read trial info from the task file. Load any stimuli needed.
- ``display_instructions()``: Show task-specific instructions. Override only if the default instructions don't apply.
- ``run_trial(trial)``: Run a single trial. Display stimuli, collect responses, return the trial data.

Useful methods from the ``Task`` parent class:

- ``wait_response()``: Wait for a button press and return the response.
- ``display_trial_feedback()``: Show green/red fixation cross for correct/incorrect.
- ``screen_quit()``: Check for escape key to quit the experiment.

3. Implement the task file class
---------------------------------
Add a new class to ``MultiTaskBattery/task_file.py`` that inherits from ``TaskFile``. You need to implement:

- ``__init__()``: Call ``super().__init__(const)`` and set ``self.name`` to your task name (must match the ``name`` in ``task_table.tsv``).
- ``make_task_file()``: Generate trial-level ``.tsv`` files with columns like ``stim``, ``trial_dur``, ``iti_dur``, ``start_time``, ``end_time``, etc.

If your task generates random stimuli (no fixed stimulus file), add it to the ``tasks_without_run_number`` list in ``MultiTaskBattery/utils.py``.

4. Add stimuli (if needed)
--------------------------
If your task uses stimulus files (images, audio, video), add them to ``stimuli/<task_name>/``.

5. Add a documentation image (optional)
----------------------------------------
Drop a screenshot of your task as ``docs/images/<task_name>.png``. It will automatically appear on the task descriptions page. For multiple images use ``<task_name>_2.png``, ``<task_name>_3.png``, etc.

6. Add task details
--------------------
Add an entry for your task in ``MultiTaskBattery/task_details.json``. This provides a detailed description and documents the parameters of ``make_task_file`` on the task descriptions page. The key must match the task ``name`` from ``task_table.tsv``.

Tips for the detailed description:

- Describe what the participant sees and does on each trial.
- Mention the expected mental processes or brain regions that this task is designed to activate (e.g., "targets the language network").
- If your task has conditions, describe what each condition involves and how it differs from the others.

For example, the ``demand_grid`` entry:

.. code-block:: json

    {
        "demand_grid": {
            "detailed_description": "Participants see a sequence of boxes lighting up on a grid. They must identify the correct pattern from two options (original vs. modified).",
            "task_file_parameters": {
                "grid_size": {
                    "type": "tuple",
                    "default": "(3, 4)",
                    "description": "Size of the grid (rows, cols)."
                },
                "num_steps": {
                    "type": "int",
                    "default": "3",
                    "description": "Number of steps in the sequence shown to the participant."
                },
                "trial_dur": {
                    "type": "float",
                    "default": "7",
                    "description": "Duration of each trial in seconds."
                }
            }
        }
    }

The detailed description appears below the short description on the :ref:`task descriptions <task_descriptions>` page, and the parameters appear in a collapsible table.

7. Test
-------
Add your task to an experiment's ``make_files.py``, generate the files, and run it to verify everything works.

Submitting your task
--------------------
To contribute your task back to the repository:

1. Fork the repository on GitHub
2. Create a branch for your task (e.g. ``add-my-new-task``)
3. Make your changes (steps 1-6 above)
4. Push and open a pull request against ``main``
