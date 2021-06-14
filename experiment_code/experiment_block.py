# import libraries
import os
import pandas as pd
import numpy as np
import math
import glob

from psychopy import visual, core, event, gui # data, logging

import experiment_code.constants as consts
from experiment_code.task_blocks import TASK_MAP
from experiment_code.ttl import ttl

class Experiment:
    """
    A general class with attributes common to experiments
    """

    def __init__(self, exp_name, behav_trianing, run_number, subj_id, ttl_flag):
        """
        exp_name        -   name of the experiment. Examples: 'mdtb_localizer', 'pontine_7T'
        run_number      -   run number
        subj_id         -   id assigned to the subject
        ttl_flag        -   flag determining whether to wait for ttl pulse or not
        behav_training  -   flag determining whether the study is done outside of the scanner or not
        """

        self.exp_name   = exp_name   
        self.run_number = run_number 
        self.subj_id    = subj_id    
        self.ttl_flag   = ttl_flag   

        # determine the name of the run file to be used
        self.run_name = f"run_{self.run_number:02}.csv"

        # if it's behavioral training then use the files under behavioral
        if behav_trianing:
            self.study_name = 'behavioral'
        else:
            self.study_name = 'fmri'
    
    def get_runfile_info(self):
        #
        """
        gets info for the run
        Returns:
            run_info(dict)   -   a dictionary with the info for the run. contains keys:
                run_file(pandas dataframe) : the opened run file in the form of a pandas dataframe
                run_num(int)               : an integer representing the run number (get from the run filename)
                task_nums(numpy array)    : a numpy array with the number of target files in each run file
        """
        self.run_info = {} # a dictionary with all the info for the run 
        # load run file
        self.run_file_path = consts.run_dir / self.study_name / self.run_name
        self.run_info['run_file'] = pd.read_csv(consts.run_dir / self.study_name / self.run_name)

        # get run num
        self.run_info['run_num'] = self.run_number

        # get number of target files in each runfile
        self.run_info['task_nums'] = np.arange(len(self.run_info['run_file']))

        return self.run_info
    
    def start_timer(self):
        """
        starts the timer for the experiment (for behavioral study)
        Returns:
            timer_info(dict)    -   a dictionary with all the info for the timer. keys are:
            global_clock : the clock from psychopy?
            t0           : the start time
        """
        #initialize a dictionary with timer info
        self.timer_info = {}

        # wait for ttl pulse or not?
        if self.ttl_flag: # if true then wait
            print(f"waiting for the TTL pulse")
            ttl.reset()
            while ttl.count <= 0:
                ttl.check()
            print(f"Received TTL pulse")
            # get the ttl clock
            self.timer_info['global_clock'] = ttl.clock
        else:
            self.timer_info['global_clock'] = core.Clock()
        
        self.timer_info['t0'] = self.timer_info['global_clock'].getTime()

        return self.timer_info

    def get_targetfile_info(self, b):
        """
        gets the target file information of the task b
        Args:
            b(int)                      -   task number
        Returns:
            target_taskInfo(dict)  -   a dictionary containing target file info for the current task with keys:
                task_name     : task name
                task_num      : task number
                target_file   : target csv file opened as a pandas dataframe
                task_endTime   : end time of the task run 
                task_startTime : start time of the task run
                instruct_dur  : duration of instruction for the task
        """
        target_binfo = {}
        # what's the task in this target file?
        target_binfo['task_name'] = self.run_info['run_file']['task_name'][b]

        # what number is this target file?
        target_binfo['target_num'] = self.run_info['run_file']['target_num'][b]
        if not math.isnan(target_binfo['target_num']):
            target_binfo['target_num'] = (f"{int(target_binfo['target_num']):02d}")

        # load target file
        target_binfo['target_file'] = pd.read_csv(consts.target_dir / self.study_name / target_binfo['task_name'] / self.run_info['run_file']['target_file'][b])

        # get end time of task
        target_binfo['task_endTime'] = self.run_info['run_file']['end_time'][b]

        # get start time of task
        target_binfo['task_startTime'] = self.run_info['run_file']['start_time'][b]

        # get instruct dur
        target_binfo['instruct_dur'] = self.run_info['run_file']['instruct_dur'][b]

        return target_binfo

    def check_runfile_results(self):
        """
        Checks if a file for behavioral data of the current run already exists
        Args:
            experiment_info(dict)   -   a dictionary with all the info for the experiment (after user inputs info in the GUI)
        Returns:
            run_iter    -   how many times this run has been run:)
        """

        self.run_dir = consts.raw_dir / self.study_name / 'raw' / self.subj_id / f"{self.study_name}_{self.subj_id}.csv"
        if os.path.isfile(self.run_dir):
            # load in run_file results if they exist 
            self.run_file_results = pd.read_csv(self.run_dir)
            if len(self.run_file_results.query(f'run_name=="{self.run_name}"')) > 0:
                current_iter = self.run_file_results.query(f'run_name=="{self.run_name}"')['run_iter'].max() # how many times has this run_file been executed?
                self.run_iter = current_iter+1
            else:
                self.run_iter = 1 
        else:
            self.run_iter = 1
            self.run_file_results = pd.DataFrame()

        return
    
    def set_runfile_results(self, all_run_response, save = True):
        """
        gets the behavioral results of the current run and returns a dataframe to be saved
        Args:
            all_run_response(list)    -   list of dictionaries with behavioral results of the current run
            save(bool)                -   if True, saves the run results. Default: True
        Returns:
            run_file_results(pandas dataframe)    -   a dataframe containing behavioral results
        """

        # save run results 
        new_df = pd.concat([self.run_info['run_file'], pd.DataFrame.from_records(all_run_response)], axis=1)

        # check if a file with results for the current run already exists
        self.check_runfile_results()
        self.run_file_results = pd.concat([self.run_file_results, new_df], axis=0, sort=False)

        # save the run results if save is True
        if save:
            self.run_file_results.to_csv(self.run_dir, index=None, header=True)

        return self.run_file_results


def get_experiment_info(debug = True):
    """
    Gets the experiment information.
    Info entered here will be used to create experiment
    Args:
    debug (bool)    -   if True, uses default names and info for testing, else, a dialogue box will pop up 
    """
    if not debug:
        # a dialog box pops up so you can enter info
        #Set up input box
        inputDlg = gui.Dlg(title = "Run Experiment")
        
        inputDlg.addField('Enter Experiment name:') # name of the experiment
        inputDlg.addField('Enter Subject ID:')      # id assigned to the subject
        inputDlg.addField('Enter Run Number (int):')      # run number (int)
        inputDlg.addField('Is it a training session?', initial = True) 
        inputDlg.addField('Wait for TTL pulse?', initial = False) # a checkbox 

        inputDlg.show()

        # # record input variables
        experiment_info = {}
        if gui.OK:
            experiment_info['exp_name']       = inputDlg.data[0]
            experiment_info['subj_id']        = inputDlg.data[1]
            experiment_info['run_number']     = inputDlg.data[2]
            experiment_info['behav_training'] = inputDlg.data[3]

            # ttl flag that will be used to determine whether the program waits for ttl pulse or not
            experiment_info['ttl_flag'] = inputDlg.data[4]
        else:
            sys.exit()

    else: 
        # uses a toy example with toy input 
        ### Values can be changed manually here
        experiment_info = {}
        experiment_info['exp_name']       = 'test'
        experiment_info['subj_id']        = 'test'
        experiment_info['run_number']     = int(input("enter the run number: ")) # change this to check other runs
        experiment_info['behav_training'] = True # change this to False to check scanning files
        experiment_info['ttl_flag']       = True # initially set this to False to check the code without ttl pulse syncing


    return experiment_info

def show_scoreboard(subj_dir, taskObjs, screen):
    """
    Presents a score board in the end of the run
    Args:
        subj_dir(str:path)    -   directory where run results of a subject is stored
        taskObjs(list)        -   a list containing task objects
        screen                -   screen object for display
    """
    # loop over task objects and get the feedback
    feedback_all = []
    for obj in taskObjs:

        # get the task name
        t_name = obj.name
        
        # discard rest. There are no specific feedback for rest and it can be excluded from the final scoreboard
        #if t_name != 'rest':
        if t_name not in ['rest', 'verb_generation', 'romance_movie', 'act_obs_knots']:

            # get the response dataframe saved for the task
            dataframe = pd.read_csv(glob.glob(os.path.join(subj_dir , f'*{t_name}*'))[0])

            # get the feedback dictionary for the task
            feedback = obj.get_task_feedback(dataframe, obj.feedback_type)

            # get the corresponding text for the feedback and append it to the overal list 
            feedback_text = f'{t_name}\n\nCurrent score: {feedback["curr"]} {feedback["measure"]}\n\nPrevious score: {feedback["prev"]} {feedback["measure"]}'
            feedback_all.append(feedback_text)

    # display feedback table at the end of the run
    ## position where the feedback for each task will be shown
    positions = [(-9, -6), (0, -6), (9, -6),
                (-9, 3), (0, 3), (9, 3)]
    for position, feedback in zip(positions, feedback_all):
        scoreboard = visual.TextStim(screen.window, text = feedback, color = [-1, -1, -1], pos = position, height = 0.5)
        scoreboard.draw()

    screen.window.flip()

    event.waitKeys()

    return

    
