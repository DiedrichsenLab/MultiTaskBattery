import experiment_code.make_target as mt
import experiment_code.utils as ut
import constants as const
import numpy as np

""" This is an example script to make the run files and trial files for an experiment"""

tasks = ['n_back','rest','verb_generation','tongue_movement', 'flexion_extension'] # ,'social_prediction','verb_generation'

#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.target_dir / task)

for r in range(1,9):
    tfiles = [f'n_back_{r:02d}.tsv','rest_30s.tsv',f'verb_generation_{r:02d}.tsv', f'tongue_movement_{r:02d}.tsv', f'flexion_extension_{r:02d}.tsv'] # ,f'social_prediction_{r:02d}.tsv',f'verb_generation_{r:02d}.tsv'
    T  = mt.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
        cl = mt.get_task_class(task)
        myTask = getattr(mt,cl)(const)
        myTask.make_trial_file(file_name = tfile)

