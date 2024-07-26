# created 2024: Caro Nettekoven
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import constants as const
import shutil

""" Script to nmake the social cognition run files and trial files"""

# this is a full list of the tasks that will be run for this localizer, do not change this list
tasks = ['theory_of_mind','n_back', 'auditory_narrative', 'spatial_navigation']



# make 30 subject numbers
subj_list = ['sub-04','sub-05','sub-06','sub-07','sub-08','sub-09','sub-10',\
             'sub-11','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20',\
            'sub-21','sub-22','sub-23','sub-24','sub-25','sub-26','sub-27','sub-28','sub-29','sub-30']


#  check if dirs for the tasks and runs exist, if not, make them
ut.dircheck(const.run_dir)
for task in tasks:
    ut.dircheck(const.task_dir / task)


for subj in subj_list:
    for r in range(1,9):
        valid_run_file = False    
        while not valid_run_file: # this is necessary to make sure that the run files are valid (i.e. no auditory narrative adjacent to intact or degraded passage)
            # making the run files
            tfiles = [f'{subj}_{task}_{r:02d}.tsv' for task in tasks]
            T  = tf.make_run_file(tasks,tfiles)

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
                T['start_time'] += 5
                T['end_time'] += 5
                # add 10 seconds to the end_time of the last task in the run
                T.loc[T.index[-1], 'end_time'] += 10
                T.to_csv(const.run_dir / f'{subj}_run_{r:02d}.tsv', sep='\t', index=False)
                break  # Valid run file found, exit the while loop
            
        # rewrite task args but with empty dict for all tasks
        task_args = {task: {} for task in tasks}
        
        # Define tasks that need run_number as an argument
        for task in ['theory_of_mind','auditory_narrative', 'spatial_navigation']:
            task_args[task].update({'run_number': r})
        
        # Define or update task specific arguments that depend on the run number (this is only for tasks that will require one hand presses)
        if r % 2 == 0:
            responses = [1, 2]
        else:
            responses = [3, 4]

        for task in ['theory_of_mind', 'n_back']:
            task_args[task].update({'responses': responses})


        # for each of the runs, make task files
        for task,tfile in zip(tasks, tfiles):
            cl = tf.get_task_class(task)
            myTask = getattr(tf,cl)(const)
            myTask.make_task_file(file_name = tfile, **task_args.get(task, {}))


        
    