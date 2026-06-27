import inspect
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const

# (task, condition) pairs. condition is None for single-condition tasks;
# verb_generation is split into its two conditions ('read' / 'generate'),
# each generated as its own block.
blocks = [('finger_sequence', None),
          ('n_back', None),
          ('demand_grid', None),
          ('auditory_narrative', None),
          ('sentence_reading', None),
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

# Generate run files that specify the order and duration of task blocks
for r in range(1, num_runs + 1):
    tasks = [task for task, cond in blocks]
    tfiles = [f'{task}_{cond}_{r:02d}.tsv' if cond else f'{task}_{r:02d}.tsv'
              for task, cond in blocks]
    T = tf.make_run_file(tasks, tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv', sep='\t', index=False)

    # Generate a task_file for each block in each run that specifies the trial information
    for (task, cond), tfile in zip(blocks, tfiles):
        row = T.loc[T['task_file'] == tfile].iloc[0]
        cl = tf.get_task_class(task)
        myTask = getattr(tf, cl)(const)

        # Pass condition only when the block has one, and run_number only if
        # the task's make_task_file accepts it.
        args = {'task_dur': row['task_dur']}
        if cond is not None:
            args['condition'] = cond
        if 'run_number' in inspect.signature(myTask.make_task_file).parameters:
            args['run_number'] = r

        # Make task file
        myTask.make_task_file(file_name=tfile, **args)
