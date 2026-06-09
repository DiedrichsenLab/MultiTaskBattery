import MultiTaskBattery.task_file as tf
import task_olive as to
import MultiTaskBattery.utils as ut
import constants as const

#tasks = ['rest_surprise_sound_images','finger_sequence_surprise','temp_deviant','theory_of_mind_diff_reward','demand_grid_easy_diff']

#tasks = ['rest_surprise_sound_images', 'finger_sequence_surprise','temp_deviant', 'theory_of_mind_diff_reward', 'demand_grid_easy_diff',
 #        'verb_generation','spatial_navigation','rest','movie','faux_pas','action_observation','tongue_movement','visual_search']

tasks = ['audio_test']

num_runs = 1  # Number of imaging runs

# Ensure task and run directories exist
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

# Generate run and task files
for r in range(1,2):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T = tf.make_run_file(tasks, tfiles, offset=3, exp_dir=const.exp_dir)
    T.loc[T.index[-1], 'end_time'] += 8
    T.to_csv(const.run_dir / f'audio_test_{r:02d}.tsv', sep='\t', index=False)

    # Generate a target file for each run
    for task, tfile in zip(tasks, tfiles):
        cl = tf.get_task_class(task, exp_dir=const.exp_dir)
        if hasattr(to, cl + 'File'):
            myTask = getattr(to, cl + 'File')(const)
        else:
            myTask = getattr(tf, cl)(const)
        # Add run number if necessary
        args = {}
        if myTask.name not in ut.tasks_without_run_number:
            args.update({'run_number': r})

        # Make task file
        myTask.make_task_file(file_name=tfile, **args)
         