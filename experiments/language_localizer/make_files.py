# created 2023: Bassel Arafat, Jorn Diedrichsen
import MultiTaskBattery.task_file as mt
import MultiTaskBattery.utils as ut
import constants as const

""" This is an example script to make the run files and trial files for an experiment"""

# all tasks that are going to be used in the experiment
# tasks = ['demand_grid','theory_of_mind','verb_generation','degraded_passage','intact_passage',\
#          'action_observation','rest','n_back','romance_movie','sentence_reading','nonword_reading','oddball',\
#         'auditory_narrative','tongue_movement','spatial_navigation']

tasks = ['demand_grid','theory_of_mind','verb_generation','degraded_passage','intact_passage',\
         'action_observation','rest','n_back','romance_movie','sentence_reading','nonword_reading','oddball',\
        'auditory_narrative','tongue_movement','spatial_navigation']

#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)

for r in range(1,9):
    # making the run files
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T  = mt.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    # define task specific arguments
    task_args = {'demand_grid': {}, 'theory_of_mind': {'run_number' : r}, 'verb_generation': {}, 'degraded_passage': {'run_number' : r}, 'intact_passage': {'run_number' : r},\
         'action_observation': {'run_number' : r}, 'rest': {}, 'n_back': {}, 'romance_movie': {'run_number' : r}, 'sentence_reading': {'run_number' : r}, 'nonword_reading': {'run_number' : r}, 'oddball': {},\
        'auditory_narrative': {'run_number' : r}, 'tongue_movement': {}, 'spatial_navigation': {'run_number' : r}}
    
    # Define or update task specific arguments that depend on the run number
    if r % 2 == 0:
        responses = [1, 2]
    else:
        responses = [3, 4]

    for task in ['demand_grid', 'theory_of_mind', 'n_back', 'oddball']:
        task_args[task].update({'responses': responses})


    # for each of the runs, make task files
    for task,tfile in zip(tasks, tfiles):
        cl = mt.get_task_class(task)
        myTask = getattr(mt,cl)(const)
        myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))