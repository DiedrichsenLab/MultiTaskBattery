# Defines the Experiment as a class
# @ Ladan Shahshahani  - Maedbh King June 2021

# import libraries
from psychopy import visual, core, event, gui # data, logging


import experiment_code.constants as consts
import experiment_code.experiment_block as exp_block
import experiment_code.make_target as make_target
from experiment_code.screen import Screen
from experiment_code.task_blocks import TASK_MAP
from experiment_code.ttl import ttl

# 1. first go to constants.py and make sure you have made the following changes:
# experiment_name = 'pontine_7T'
# base_dir = Path(<your chosen path>).absolute()

# 2. make sure your base directory contains the following folder
# 'experiment_code'
# 'stimuli'     -   stimuli folder should contain all the stimuli for the tasks in your experiment

# 3. making sure you have all the necessary folders
# consts.dirtree()


# 4. create target files first (if already not done)
def create_target(task_list = ['visual_search', 'action_observation_knots', 'flexion_extension', 
                              'finger_sequence', 'theory_of_mind', 'n_back', 'semantic_prediction', 
                              'rest', 'romance_movie'], 
                  num_runs = 8):

    """
    makes target and session files for the pontine experiment.
    Args:
        task_list (list)    -   list containing task names to be included in the experiment
        num_runs (int)      -   number of runs to be created for the task
    """
    ## behavioral
    make_target.make_files(task_list = task_list, study_name='behavioral', num_runs = num_runs)
    ## fmri
    make_target.make_files(task_list = task_list, study_name='fmri', num_runs = num_runs)

# 5. run the experiment.
## change debug to False once you are sure everything is debugged 
def run(debug = True):
    

    # 1. get experiment information
    exp_info = exp_block.set_experiment_info(debug = debug)

    ### printing some info to the screen for developer
    print(f"***** you are in debug mode {debug}")
    print(f"***** experiment name: {exp_info['exp_name']}")
    print(f"***** subject id: {exp_info['subj_id']}")
    print(f"***** run number: {exp_info['run_number']}")
    print(f"    * if in debugging and want to check another run, go back to experiment_block.get_experiment_info and change the run number manually")
    
    if exp_info['ttl_flag']:
        print(f"***** waiting for the TTL pulse! press '5'")

    # 2. create a class for the experiment
    Custom_Exp = exp_block.Experiment(exp_info['exp_name'], exp_info['behav_training'], 
                                      exp_info['run_number'], exp_info['subj_id'], 
                                      exp_info['ttl_flag'], exp_info['eyetrack_flag'])


    # 3. get the run file info: creates self.run_info
    run_info = Custom_Exp.get_runfile_info()

    # 4. make subject folder in data/raw/<subj_id>
    subj_dir = consts.raw_dir/ Custom_Exp.study_name / 'raw' / exp_info['subj_id']
    consts.dircheck(subj_dir)

    Custom_Exp.check_runfile_results()

    # 5. open screen and display fixation cross
    exp_screen = Screen()

    # 6. timer stuff!
    ## start the timer. Needs to know whether the experimenter has chosen to wait for ttl pulse 
    timer_info = Custom_Exp.start_timer()

    # 7. initialize a list for responses
    all_run_response = []

    # 8. loop over tasks in the run file
    taskObj_list  = [] # an empty list. Task classes will be appended to this list
    for b in run_info['task_nums']:

        # 8.1 get target info
        target_binfo = Custom_Exp.get_targetfile_info(b)

        # 8.2 get the real strat time for each task 
        ## for debugging make sure that this is at about the start_time specified in the run file
        real_start_time = timer_info['global_clock'].getTime() - timer_info['t0']
        print(f"real_start_time:{real_start_time} == start_time: {target_binfo['task_startTime']}????") # for debugging purposes!

        # 8.2.1 collect ttl time and counter
        # if ttl_flag:
        #     ttl_time  = ttl.time - timer_info['t0']
        #     ttl_count = ttl.count
        # else:
        #     ttl_time  = 0
        #     ttl_count = 0

        # 8.3 get the task object and append it to a list
        TaskName = TASK_MAP[target_binfo['task_name']]

        Task_Block  = TaskName(screen = exp_screen, 
                              target_file = target_binfo['target_file'], 
                              run_end  = target_binfo['task_endTime'], task_name = target_binfo['task_name'],  
                              study_name = Custom_Exp.study_name, target_num = target_binfo['target_num'], 
                              ttl_flag = Custom_Exp.ttl_flag)

        taskObj_list.append(Task_Block)

        # 8.4 wait till it's time to start the task
        while timer_info['global_clock'].getTime() - timer_info['t0'] <= target_binfo['task_startTime']:
            if exp_info['ttl_flag']:
                ttl.check()
            else:
                pass
    
        # 8.5 get the instruction text for the task and display it
        Task_Block.display_instructions()

        # 8.6 wait for a time period equal to instruction duration
        wait_time = target_binfo['task_startTime'] + target_binfo['instruct_dur']
        while timer_info['global_clock'].getTime() - timer_info['t0'] <= wait_time: # timed presentation of the instruction
            if exp_info['ttl_flag']:
                ttl.check()
            else:
                pass

        # 8.7.1 run task and collect feedback
        new_resp_df = Task_Block.run()

        # 8.7.2 adding run information to response dataframe
        new_resp_df['run_name'] = Custom_Exp.run_name
        new_resp_df['run_iter'] = Custom_Exp.run_iter
        new_resp_df['run_num']  = Custom_Exp.run_number
        # 8.7.3 get the response dataframe and save it
        fpath = consts.raw_dir / Custom_Exp.study_name/ 'raw' / exp_info['subj_id'] / f"{Custom_Exp.study_name}_{exp_info['subj_id']}_{target_binfo['task_name']}.csv"
        Task_Block.save_task_response(new_resp_df, fpath)
        # save_response(new_resp_df, exp_info['study_name'], exp_info['subj_id'], target_binfo['task_name'])

        # 8.8 log results
        # collect real_end_time for each task
        all_run_response.append({
            'real_start_time': real_start_time,
            'real_end_time': (timer_info['global_clock'].getTime() - timer_info['t0']),
            # 'ttl_counter': ttl_count,
            # 'ttl_time': ttl_time,
            'run_name': Custom_Exp.run_name,
            'task_idx': b+1,
            'run_iter': Custom_Exp.run_iter,
            'run_num': run_info['run_num'],
        })

        # 8.9 wait till it's time to end the task
        while timer_info['global_clock'].getTime() - timer_info['t0'] <= target_binfo['task_endTime']: # timed presentation
            if exp_info['ttl_flag']:
                ttl.check()
            else:
                pass
    
    # 9.1 get the run result as a dataframe
    df_run_results = Custom_Exp.set_runfile_results(all_run_response, save = True)

    # 10. present feedback from all tasks on screen 
    Custom_Exp.show_scoreboard(taskObj_list, exp_screen)

    # 11. end experiment
    # end_exper_text = f"End of run {self.run_num}\n\nTake a break!"
    end_exper_text = f"End of run\n\nTake a break!"
    end_experiment = visual.TextStim(exp_screen.window, text=end_exper_text, color=[-1, -1, -1])
    end_experiment.draw()
    exp_screen.window.flip()

    # waits for a key press to end the experiment
    event.waitKeys()
    # quit screen and exit
    exp_screen.window.close()
    core.quit()

    return