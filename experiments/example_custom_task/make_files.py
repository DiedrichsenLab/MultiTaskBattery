"""Generate run files and task files for example_custom_task.

Mixes built-in MTB tasks (n_back, rest) with two custom tasks defined locally
in my_tasks.py.

Class lookups go through ut.get_task_class (runtime) and
ut.get_task_file_class (file generation). Both consult const.task_modules
first, then fall back to the shared MultiTaskBattery package. The
'<class>File' suffix convention for custom TaskFile classes is encapsulated
inside ut.get_task_file_class — make_files.py doesn't need to know about it.

Whether to pass run_number is decided by inspecting each task's
make_task_file signature.
"""

import inspect
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const

# Each block is (task_name, kwargs). The kwargs are passed straight to that
# task's make_task_file() — e.g. {'condition': 'sentences'}, {'n_back': 3}.
# A 'condition' kwarg is also used to name the task file, so two blocks of the
# same task in different conditions get their own files. Empty {} = defaults.
blocks = [
    ('n_back',      {}),
    ('rest',        {}),
    ('silent_word', {}),   # custom task, defined in my_tasks.py
    ('odd_even',    {}),   # custom task, defined in my_tasks.py
]

num_runs = 3  # Number of imaging runs

# Ensure task and run directories exist
ut.dircheck(const.run_dir)
for task, _ in blocks:
    ut.dircheck(const.task_dir / task)

# Generate the run file and each task file, for every run
for r in range(1, num_runs + 1):
    tasks  = [task for task, _ in blocks]
    # Name each task file after its condition, if any. A list condition is
    # joined with '-' (e.g. ['read', 'generate'] -> read-generate), so repeated
    # blocks of the same task in different conditions get distinct files.
    tfiles = []
    for task, kw in blocks:
        cond = kw.get('condition')
        if isinstance(cond, (list, tuple)):
            tag = '-'.join(map(str, cond))
        else:
            tag = str(cond) if cond is not None else ''
        tfiles.append(f"{task}_{tag}_{r:02d}.tsv" if tag else f"{task}_{r:02d}.tsv")
    assert len(set(tfiles)) == len(tfiles), (
        "Two blocks map to the same task file — give repeated blocks of the "
        "same task distinct 'condition' values so their task files differ.")

    # Pass exp_dir so any local task_table.tsv (for custom tasks) is merged
    # with the framework's table.
    T = tf.make_run_file(tasks, tfiles, exp_dir=const.exp_dir)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv', sep='\t', index=False)

    # Generate a task file for each block that specifies the trial information
    for (task, kw), tfile in zip(blocks, tfiles):
        row = T.loc[T['task_file'] == tfile].iloc[0]

        # Resolve the TaskFile generator: a custom '<Class>File' in
        # const.task_modules if present, otherwise the built-in one.
        cl = tf.get_task_class(task, exp_dir=const.exp_dir)
        myTask = ut.get_task_file_class(const, cl)(const)

        # Start from the block's task_dur, add its kwargs, and pass run_number
        # only if this task's make_task_file declares it.
        args = {'task_dur': row['task_dur'], **kw}
        if 'run_number' in inspect.signature(myTask.make_task_file).parameters:
            args['run_number'] = r

        myTask.make_task_file(file_name=tfile, **args)
