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
# tasks = ['rmet', 'rmet', 'theory_of_mind','demand_grid']
# conditions = ['rmet_age', 'rmet_emotion', None, None]
tasks = ['rmet', 'rmet', 'theory_of_mind', 'theory_of_mind', 'demand_grid']
conditions = ['rmet_age', 'rmet_emotion', 'tom_belief', 'tom_photo', None, None]
# tasks = ['theory_of_mind'] # Test if working memory default settings are working correctly
# conditions = None
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

for r in range(1,10):
    tfiles = [f'{task}_{r:02d}.tsv' for task in tasks]   
    task_args = {task: {} for task in tasks}

    # Deal with run number for tasks that need the information
    for task in tasks:
        if task not in ut.tasks_without_run_number:
            task_args[task].update({'run_number': r})
    
    # Deal with conditions
    if conditions is not None:
        # task_args now needs two fields for rmet, the rmet_age and rmet_emot
        
        for i, condition in enumerate(conditions):
            if condition is not None:
                # Deal with run files
                tfiles[i] = f'{condition}_{r:02d}.tsv'
                # Deal with task files
                # Make a copy of the task_args for the general rmet task
                task_args[condition] = task_args[tasks[i]].copy()
                # Update the stim file with the correct one
                task_args[condition].update({'condition': condition.split('_')[1]})


    T  = tf.make_run_file(tasks,tfiles, conditions=conditions)
    T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)
    

    # for each of the runs, make a target file
    for t,task_info in enumerate(zip(tasks, tfiles)):
        task, tfile = task_info
        cl = tf.get_task_class(task)
        myTask = getattr(tf,cl)(const)
        print(task)
        if conditions is not None:
            if conditions[t] is not None:
                task = conditions[t]
        myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))

    