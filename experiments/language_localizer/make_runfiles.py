# created 2023: Bassel Arafat, Jorn Diedrichsen
import MultiTaskBattery.task_file as mt
import MultiTaskBattery.utils as ut
import constants as const

""" This is an example script to make the run files and trial files for an experiment"""

# all tasks that are going to be used in the experiment
# tasks = ['demand_grid','theory_of_mind','verb_generation','degraded_passage','intact_passage',\
#          'action_observation','rest','n_back','romance_movie','sentence_reading','nonword_reading','oddball',\
#         'auditory_narrative','tongue_movement','spatial_navigation']

tasks = ['theory_of_mind']

#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

for r in range(1,9):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T  = mt.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
        cl = mt.get_task_class(task)
        myTask = getattr(mt,cl)(const)

        if task == 'n_back' or task == 'rest' or task == 'verb_generation' or task == 'tongue_movement' or task == 'spatial_navigation' or task == 'oddball' or task == 'demand_grid':
            myTask.make_task_file(file_name = tfile)
        else:
            myTask.make_task_file(file_name = tfile, run_number = r)