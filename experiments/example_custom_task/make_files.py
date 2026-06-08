"""Generate run files and task files for example_custom_task.

Mixes built-in MTB tasks (n_back, rest) with two custom tasks defined locally
in my_tasks.py.

Class lookups go through ut.get_task_class (runtime) and
ut.get_task_file_class (file generation). Both consult const.task_modules
first, then fall back to the shared MultiTaskBattery package. The
'<class>File' suffix convention for custom TaskFile classes is encapsulated
inside ut.get_task_file_class — make_files.py doesn't need to know about it.

Whether to pass run_number is decided by inspecting each task's
make_task_file signature, so custom tasks don't need to be added to
ut.tasks_without_run_number.
"""

import inspect
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const


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
        TaskFileCls = ut.get_task_file_class(const, cl)
        myTask = TaskFileCls(const)

        args = {}
        if 'run_number' in inspect.signature(myTask.make_task_file).parameters:
            args['run_number'] = r

        myTask.make_task_file(file_name=tfile, **args)
