Implementing new tasks
======================

There are two scenarios for adding a task to MultiTaskBattery:

1. **Adding a task locally to your experiment.** The task lives only in
   your experiment folder and is registered via ``task_modules`` in your
   ``constants.py``. The shared package is unchanged. This is the right
   path for almost everyone — see ``experiments/example_custom_task`` for
   a working reference.

2. **Contributing a task to the shared library.** The task is added to
   ``MultiTaskBattery/task_blocks.py``, ``task_file.py``, and
   ``task_table.tsv``, documented in ``task_details.json``, and
   submitted via pull request. Use this path only when you want others
   outside your project to use the task too.

Adding a task locally to your experiment
----------------------------------------

Recommended layout: both the runtime ``Task`` class and the file-generator
``TaskFile`` class for one task live in a single module in your experiment
folder.

1. Create a local module
^^^^^^^^^^^^^^^^^^^^^^^^
Add a new ``.py`` file (e.g. ``my_tasks.py``) in your experiment folder
that defines both classes:

.. code-block:: python

    from MultiTaskBattery.task_blocks import Task
    from MultiTaskBattery.task_file import TaskFile
    from psychopy import visual, event
    import pandas as pd
    import numpy as np

    class MyTask(Task):
        def __init__(self, info, screen, ttl_clock, const, subj_id):
            super().__init__(info, screen, ttl_clock, const, subj_id)
            self.feedback_type = 'acc+rt'   # or 'none', 'acc', 'rt'

        def init_task(self):
            ...  # read trial info, load stimuli

        def display_instructions(self):
            ...  # task-specific instructions

        def run_trial(self, trial):
            ...  # display stimulus, collect response, return trial

    class MyTaskFile(TaskFile):
        def __init__(self, const):
            super().__init__(const)
            self.name = 'my_task'  # must match the row in task_table.tsv

        def make_task_file(self, ..., file_name=None):
            ...  # generate trial-level rows, write tsv

Methods to implement on the ``Task`` subclass:

- ``init_task()``: Read the task's trial-info TSV into ``self.trial_info``.
  Load any stimuli into memory.
- ``display_instructions()``: Show task-specific instructions on the screen.
- ``run_trial(trial)``: Run a single trial — display stimuli, collect responses,
  return the trial row with any added columns.

Useful methods inherited from the ``Task`` parent:

- ``wait_response(start_time, max_wait_time)``: Wait for a button press
  and return ``(key, rt)``.
- ``display_trial_feedback(give_feedback, correct)``: Show a green check or
  red cross based on correctness.
- ``screen_quit()``: Check for the escape key to quit the experiment.

``feedback_type`` controls the end-of-run scoreboard: ``'none'``, ``'acc'``,
``'rt'``, or ``'acc+rt'``.

If your task generates random stimuli (no fixed stimulus file per run),
omit ``run_number`` from the ``make_task_file`` signature — ``make_files.py``
inspects the signature to decide whether to pass it.

2. Add a row to a local task_table.tsv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a ``task_table.tsv`` file in your experiment folder with a single
tab-separated row for your task (same columns as the shared table):

.. code-block:: text

    name	task_class	descriptive_name	code
    my_task	MyTask	my_task	mytsk

The local table is merged with the shared one automatically when
``make_files.py`` passes ``exp_dir=const.exp_dir`` to ``tf.make_run_file``
and ``tf.get_task_class`` (the example ``make_files.py`` already does this).

3. Register your module in constants.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Import your local module and add it to ``task_modules``:

.. code-block:: python

    import my_tasks
    task_modules = [my_tasks]

At runtime, ``ut.get_task_class`` walks ``task_modules`` first and falls
back to the shared package. At file-generation time,
``ut.get_task_file_class`` does the same — and appends ``'File'``
automatically when searching local modules — so ``make_files.py`` needs
zero changes for new tasks.

4. Add stimuli (if your task uses them)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Stimulus files (images, audio, video) live under
``stimuli/<task_name>/`` at the repository root. If your task generates
stimuli procedurally, skip this step.

5. Test
^^^^^^^
Add your task to the ``blocks`` list in your experiment's
``make_files.py`` (as ``('my_task', None)``, or with a condition),
generate the run and task files, and run ``run.py``.
``experiments/example_custom_task`` is the reference for a working
custom-task setup.

Contributing a task to the shared library
-----------------------------------------

If you want your task to be included in MultiTaskBattery so it is
available to other users, do everything from the local section above,
*plus* the extra steps below.

1. Move the classes into the shared package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the shared package the runtime and file-generator classes live in
separate modules and reuse the same bare name (no ``File`` suffix):

- Move the ``Task`` subclass into ``MultiTaskBattery/task_blocks.py``.
- Move the ``TaskFile`` subclass into ``MultiTaskBattery/task_file.py``,
  dropping the ``File`` suffix from its class name.
- Move the row from your local ``task_table.tsv`` into
  ``MultiTaskBattery/task_table.tsv``.

2. Add task details
^^^^^^^^^^^^^^^^^^^
Add an entry for your task in ``MultiTaskBattery/task_details.json``.
The key must match the task ``name`` from ``task_table.tsv``. Each
entry should include:

- ``short_description``: a brief one-line summary of the task.
- ``detailed_description``: a longer description of what the task involves.
- ``recorded_metrics``: ``Accuracy + RT``, ``Accuracy``, ``RT``, or ``None``.
- ``conditions``: comma-separated list of conditions (omit if none).
- ``reference``: academic citation (omit if none).
- ``task_file_parameters``: documents the parameters of ``make_task_file``.

Tips for the detailed description:

- Describe what the participant sees and does on each trial.
- Mention the expected mental processes or brain regions that the task
  is designed to activate (e.g., "targets the language network").
- If your task has conditions, describe what each involves and how
  they differ.

For example, the ``demand_grid`` entry:

.. code-block:: json

    {
        "demand_grid": {
            "short_description": "2AFC spatial working memory task on a grid.",
            "detailed_description": "Participants see a sequence of boxes lighting up on a grid...",
            "recorded_metrics": "Accuracy + RT",
            "reference": "Fedorenko et al. (2013)...",
            "task_file_parameters": {
                "grid_size": {
                    "type": "tuple",
                    "default": "(3, 4)",
                    "description": "Size of the grid (rows, cols)."
                }
            }
        }
    }

All fields appear on the :ref:`task descriptions <task_descriptions>`
page, with the parameters shown in a collapsible table.

3. Add a documentation image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Drop a screenshot of your task as ``docs/images/<task_name>.png``. It
will automatically appear on the task descriptions page. For multiple
images use ``<task_name>_2.png``, ``<task_name>_3.png``, etc. You can
also add a short demo video as ``docs/images/<task_name>.mp4`` (and
``<task_name>_2.mp4``, ...) to render an inline video player.

4. Open a pull request
^^^^^^^^^^^^^^^^^^^^^^
1. Fork the repository on GitHub.
2. Create a branch for your task (e.g. ``add-my-new-task``).
3. Make your changes (steps 1-3 above).
4. Push and open a pull request against ``main``.
