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


blocks = [('n_back', None), ('rest', None), ('silent_word', None), ('odd_even', None)]
num_runs = 3

# Ensure task and run directories exist
ut.dircheck(const.run_dir)
for task, cond in blocks:
    ut.dircheck(const.task_dir / task)

# Generate run files that specify the order and duration of task blocks
for r in range(1, num_runs + 1):
    tasks = [task for task, cond in blocks]
    tfiles = [f'{task}_{cond}_{r:02d}.tsv' if cond else f'{task}_{r:02d}.tsv' for task, cond in blocks]
    T = tf.make_run_file(tasks, tfiles, exp_dir=const.exp_dir)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv', sep='\t', index=False)

    # Generate a task_file for each task in each run that specifies the trial information
    for (task, cond), tfile in zip(blocks, tfiles):
        row = T.loc[T['task_file']==tfile].iloc[0]
        cl = tf.get_task_class(task, exp_dir=const.exp_dir)
        TaskFileCls = ut.get_task_file_class(const, cl)
        myTask = TaskFileCls(const)

        # Only pass run_number if make_task_file actually accepts it., pass in task duration as a default
        args = {'task_dur':row['task_dur']}
        if cond is not None:
            args['condition']=cond
        if 'run_number' in inspect.signature(myTask.make_task_file).parameters:
            args['run_number'] = r

        myTask.make_task_file(file_name=tfile, **args)
