# created 2023: Bassel Arafat, Jorn Diedrichsen
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const
import shutil
import pandas as pd

""" This is an example script to make the run files and trial files for an experiment"""

# this is a full list of the tasks that will be run for this localizer, do not change this list
full_tasks = ['demand_grid','theory_of_mind','verb_generation','degraded_passage','intact_passage',\
         'action_observation','rest','n_back','romance_movie','sentence_reading','nonword_reading','oddball',\
        'auditory_narrative','tongue_movement','spatial_navigation','finger_sequence']


# this is a list of the tasks running while debugging and testing different combos and will be used when the final combo is ready (having both this list and the above
#is necessary because I have the task_args list defined first then I am putting conditional statements for specific arguments,
# for the script to run, the conditional stuff needs to run and for the conditional stuff to run all tasks (full_tasks)need to be inside task_args)
running_tasks = ['demand_grid'] # adjust this list as you like to test different combos


#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in running_tasks:
    ut.dircheck(const.task_dir / task)


for r in range(1,9):
    valid_run_file = False    
    while not valid_run_file: # this is necessary to make sure that the run files are valid (i.e. no auditory narrative adjacent to intact or degraded passage)
        # making the run files
        tfiles = [f'training_{task}_{r:02d}.tsv' for task in running_tasks]
        T  = tf.make_run_file(running_tasks,tfiles)

        tasks = T['task_name'].tolist()
        valid_run_file = True

        for i in range(len(tasks) - 1):
            if tasks[i] == 'auditory_narrative' and tasks[i + 1] in ['intact_passage', 'degraded_passage']:
                valid_run_file = False
                break
            if tasks[i + 1] == 'auditory_narrative' and tasks[i] in ['intact_passage', 'degraded_passage']:
                valid_run_file = False
                break

        if not valid_run_file:
            print(f'Run {r} is not valid. Trying again...')
        else:
            print(f'Run {r} is valid. Saving run file...')

        if valid_run_file:
            # shift everything in the run file by 5 seconds
            # add 10 seconds to the end_time of the last task in the run
            T.to_csv(const.run_dir / f'training_run_{r:02d}.tsv', sep='\t', index=False)
            break  # Valid run file found, exit the while loop
        
    # rewrite task args but with empty dict for all tasks
    task_args = {task: {} for task in full_tasks}
    
    # Define tasks that need run_number as an argument
    for task in ['theory_of_mind', 'degraded_passage', 'intact_passage', 'action_observation', 'romance_movie', 'sentence_reading', 'nonword_reading', 'auditory_narrative', 'spatial_navigation']:
        task_args[task].update({'run_number': r})
    
    # Define or update task specific arguments that depend on the run number (this is only for tasks that will require one hand presses)
    if r % 2 == 0:
        responses = [1, 2]
    else:
        responses = [3, 4]

    for task in ['demand_grid', 'theory_of_mind', 'n_back', 'oddball']:
        task_args[task].update({'responses': responses})


    # This is specific to tasks that will require presses using both hand across runs
    for task in ['finger_sequence']:
        responses = [1, 2, 3, 4]
        task_args[task].update({'responses': responses})

    task_args['verb_generation'].update({'stim_file': const.stim_dir / 'verb_generation' / 'training_verb_generation.csv'})
    task_args['theory_of_mind'].update({'stim_file': const.stim_dir / 'theory_of_mind' / 'training_theory_of_mind.csv'})
    task_args['sentence_reading'].update({'stim_file': const.stim_dir / 'sentence_reading' / 'training_sentences_shuffled.csv'})
    task_args['nonword_reading'].update({'stim_file': const.stim_dir / 'nonword_reading' / 'training_nonwords_shuffled.csv'})
    task_args['action_observation'].update({'knot_names': [
                        'Ampere', 'Arbor', 'Baron', 'Belfry', 'Bramble', 'Chamois', 'Coffer', 
                        'Farthing', 'Fissure', 'Gentry', 'Henchman', 'Magnate', 'Perry', 'Phial', 'Polka', 
                        'Rosin', 'Shilling', 'Simper', 'Spangle', 'Squire', 'Vestment', 'Wampum', 'Wicket']})



    # for each of the runs, make task files
    for task,tfile in zip(running_tasks, tfiles):
        cl = tf.get_task_class(task)
        myTask = getattr(tf,cl)(const)
        myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))

        # if task is romance movie or degraded passage add 'training' to all rows in the stim column of the task file
        if task in ['romance_movie', 'degraded_passage','intact_passage','auditory_narrative']:
            # load the tsv file with pandas
            T = pd.read_csv(const.task_dir / task / tfile, sep='\t')
            T['stim'] = 'training_' + T['stim']
            T.to_csv(const.task_dir / task / tfile, sep='\t', index=False)


        