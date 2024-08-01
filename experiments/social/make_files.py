# created 2024: Caro Nettekoven
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const
import shutil

""" Script to make the social cognition run files and trial files"""
alltasks = ['theory_of_mind', 'n_back', 'action_observation', 'verb_generation',
             'romance_movie', 'rest', 'tongue_movement', 'auditory_narrative',
             'spatial_navigation', 'degraded_passage', 'sentence_reading', 
             'nonword_reading', 'oddball', 'intact_passage', 'demand_grid', 
             'finger_sequence', 'flexion_extension', 'semantic_prediction', 
             'visual_search', 'rmet']
# tasks=alltasks

# tasks = ['rmet', 'theory_of_mind','demand_grid']
tasks = [('rmet', 'rmet_age'), ('rmet', 'rmet_emot'), 'theory_of_mind','demand_grid']
# tasks = ['rmet']

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

task_list, cond_list = zip(*[(task[0], task[1]) if isinstance(task, tuple) else (task, None) for task in tasks])
for r in range(1,10):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]
    T  = tf.make_run_file(tasks,tfiles)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

    task_args = {task: {} for task in tasks}

    # Define tasks that need run_number as an argument
    for task in tasks:
        if task not in ut.tasks_without_run_number:
            task_args[task].update({'run_number': r})

    # for each of the runs, make a target file
    for task,tfile in zip(tasks, tfiles):
         cl = tf.get_task_class(task)
         myTask = getattr(tf,cl)(const)
         print(task)
         myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))

    