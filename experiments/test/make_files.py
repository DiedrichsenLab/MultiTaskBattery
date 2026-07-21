import inspect
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const

# Each block: (task_name, extra_kwargs_for_make_task_file).
blocks = [
    ('rest', {}),
]

num_runs = 2  # Number of imaging runs

# Ensure task and run directories exist
ut.dircheck(const.run_dir)
for task, _ in blocks:
    ut.dircheck(const.task_dir / task)

# Generate run files that specify the order and duration of task blocks
for r in range(1, num_runs + 1):
    tasks = [task for task, _ in blocks]
    tfiles = [f'{task}_{r:02d}.tsv' for task, _ in blocks]
    T = tf.make_run_file(tasks, tfiles, instruction_dur=1, task_dur=5)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv', sep='\t', index=False)

    # Generate a task_file for each block that specifies the trial information
    for (task, extra), tfile in zip(blocks, tfiles):
        row = T.loc[T['task_file'] == tfile].iloc[0]
        myTask = getattr(tf, tf.get_task_class(task))(const)

        # Start from the block duration, add the block's extra kwargs, and only
        # pass run_number if this task's make_task_file accepts it.
        args = {'task_dur': row['task_dur'], **extra}
        if 'run_number' in inspect.signature(myTask.make_task_file).parameters:
            args['run_number'] = r

        # Make task file
        myTask.make_task_file(file_name=tfile, **args)
