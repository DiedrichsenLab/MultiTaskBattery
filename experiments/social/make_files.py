# created 2024: Caro Nettekoven
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const
import shutil

""" Script to make the social cognition run files and trial files"""

# this is a full list of the tasks that will be run for this localizer, do not change this list
# tasks = ['rmet', 'theory_of_mind','n_back', 'spatial_navigation']
tasks = ['rmet']
tasks_without_run_number = ['n_back']
# make 30 subject numbers
subj_list = ['sub-04']


#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

for r in range(1,2):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T  = tf.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    task_args = {task: {} for task in tasks}

    # Define tasks that need run_number as an argument
    tasks_with_run_number = [task for task in tasks if task not in tasks_without_run_number]
    for task in tasks_with_run_number:
        task_args[task].update({'run_number': r})

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
         cl = tf.get_task_class(task)
         myTask = getattr(tf,cl)(const)
         myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))

    