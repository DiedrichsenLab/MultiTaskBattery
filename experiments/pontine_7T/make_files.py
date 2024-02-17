import MultiTaskBattery.task_file as mt
import MultiTaskBattery.utils as ut
import constants as const
import numpy as np

""" This is an example script to make the run files and trial files for an experiment"""

tasks = ['semantic_prediction']

#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

for r in range(1,20):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T  = mt.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    task_args = {task: {} for task in tasks}

    for task in ['semantic_prediction']:
            task_args[task].update({'run_number': r})

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
         cl = mt.get_task_class(task)
         myTask = getattr(mt,cl)(const)
         myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))
