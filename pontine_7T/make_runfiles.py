import experiment_code.make_target as mt
import experiment_code.utils as ut
import constants as const
import numpy as np

""" This is an example script to make the run files and trial files for an experiment"""

tasks = ['n_back','rest'] # ,'social_prediction','verb_generation'

for r in range(1,9):
    tfiles = [f'n_back_{r:02d}.tsv','rest_30s.tsv'] # f'social_prediction_{r:02d}.tsv',f'verb_generation_{r:02d}.tsv',
    T  = mt.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
        cl = mt.get_task_class(task)
        myTask = getattr(mt,cl)(const)
        myTask.make_trial_file(file_name = tfile)

