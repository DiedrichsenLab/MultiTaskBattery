import MultiTaskBattery.task_file as mt
import MultiTaskBattery.utils as ut
import constants as const
import numpy as np

""" This is an example script to make the run files and trial files for an experiment"""

#tasks = ['finger_sequence', 'theory_of_mind', 'semantic_prediction', 'visual_search', 'n_back', 'flexion_extension', 'romance_movie', 'rest', 'action_observation']

tasks = ['semantic_prediction', 'theory_of_mind', 'visual_search','n_back', 'finger_sequence']

#for session 1: runs 1-4 should be with the right hand
#runs 5-8 should be with the left hand 

#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

for r in range (18,20):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T  = mt.make_run_file(tasks,tfiles)
    T['start_time'] += 5
    T['end_time'] += 5
    T['hand'] = 'right' 
                # add 10 seconds to the end_time of the last task in the run
    T.loc[T.index[-1], 'end_time'] += 10
  
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    task_args = {task: {} for task in tasks}

#for semantic_prediction, romance_movie, action_observation, theory_of_mind 
    for task in ['semantic_prediction', 'theory_of_mind']:
        task_args[task].update({'run_number': r})

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
         cl = mt.get_task_class(task)
         myTask = getattr(mt,cl)(const)
         myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))
