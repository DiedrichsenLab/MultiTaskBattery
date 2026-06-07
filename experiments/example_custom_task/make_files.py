"""Generate run files and task files for example_custom_task.

Mixes built-in MTB tasks (n_back, rest) with two custom tasks defined locally
in my_tasks.py.

For each task, the class name comes from task_table.tsv (e.g.
'SilentWord'). The matching TaskFile class is found by first trying the
local my_tasks module under the name '<class>File' (e.g. 'SilentWordFile');
if not present, it falls back to the framework's task_file module under
the bare class name.

Whether to pass run_number is decided by inspecting each task's
make_task_file signature, so custom tasks don't need to be added to
ut.tasks_without_run_number.
"""

import inspect
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const
import my_tasks as mt


tasks = ['n_back', 'rest', 'silent_word', 'odd_even']
num_runs = 3

ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

for r in range(1, num_runs + 1):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T = tf.make_run_file(tasks, tfiles, exp_dir=const.exp_dir)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv', sep='\t', index=False)

    for task, tfile in zip(tasks, tfiles):
        cl = tf.get_task_class(task, exp_dir=const.exp_dir)
        try:
            myTask = getattr(mt, cl + 'File')(const)
        except AttributeError:
            myTask = getattr(tf, cl)(const)

        args = {}
        if 'run_number' in inspect.signature(myTask.make_task_file).parameters:
            args['run_number'] = r

        myTask.make_task_file(file_name=tfile, **args)
