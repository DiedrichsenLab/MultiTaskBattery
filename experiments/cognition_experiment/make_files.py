import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const

tasks = ['finger_sequence','n_back','demand_grid','auditory_narrative','sentence_reading',
            'verb_generation','action_observation','tongue_movement','theory_of_mind','rest']

num_runs = 8 # number of imaging runs

# check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

for r in range(1,11):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T  = tf.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
        cl = tf.get_task_class(task)
        myTask = getattr(tf,cl)(const)

        # check if the task needs a run number
        args={}
        if myTask.name not in ut.tasks_without_run_number:
            args.update({'run_number': r})

        # make task file
        myTask.make_task_file(file_name = tfile, **args)