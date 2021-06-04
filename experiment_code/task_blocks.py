# import libraries
from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import time
import math
import glob

from psychopy import visual, core, event, gui # data, logging
from psychopy.visual import ShapeStim

import experiment_code.constants as consts
from experiment_code.screen import Screen
from experiment_code.ttl import ttl

from ast import literal_eval


class Task:
    """
    Task: takes in inputs from run_experiment.py and methods
    some methods are universal across all tasks. Those methods are included in the super class Task.
    There are methods like display_instructions which are the same across most of the tasks. 
    There are, however, some tasks that have different instructions from the rest (like fingerSequence).
    For those tasks, a display_instructions method is defined within the corresponding class which overrides
    the universal display_instruction method.
    Each of other classes runs a unique task given input from target files and from the Task class
    (VisualSearch, SemanticPrediction, NBack, SocialPrediction, ActionObservation).
    """

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        self.screen  = screen
        self.window  = screen.window
        self.monitor = screen.monitor
        
        self.clock    = core.Clock()
        self.ttl_flag = ttl_flag

        self.target_file = target_file
        self.run_end     = run_end
        
        self.study_name  = study_name
        self.task_name   = task_name
        self.target_num  = target_num
        
        self.trial_response     = {} # an empty dictionary which will be filled with trial responses
        self.all_trial_response = [] # an empty list which will be appended with the responses from each trial

        # assign keys to hands
        ## from const, the response keys are imported first
        self.response_keys    = consts.response_keys
          
        self.key_hand_dict = {
            'right': {    # right hand
                'True':  [self.response_keys[4]], # index finger
                'False': [self.response_keys[5]],  # middle finger
                'None' : [self.response_keys[4], 
                          self.response_keys[5], 
                          self.response_keys[6], 
                          self.response_keys[7]] # four fingers from right hand
                },
            'left': {   # left hand
                'True':  [self.response_keys[2]], # Middle finger
                'False': [self.response_keys[3]],  # Index finger
                'None' : [self.response_keys[0], 
                          self.response_keys[1], 
                          self.response_keys[2], 
                          self.response_keys[3]] # four fingers from right hand
                },
            }

    # 1. get trial info (info that is shared across the tasks)
    def get_trial_info(self, trial_index):
        """
        get the information for the current trial
        task-specific trial info will be loaded inside the specific task class
        """
        self.start_time = self.target_file['start_time'][trial_index]
        # self.end_time   = self.target_file['end_time'][trial_index]
        self.trial_dur  = self.target_file['trial_dur'][trial_index]
        self.iti_dur    = self.target_file['iti_dur'][trial_index]
        self.hand       = self.target_file['hand'][trial_index]
    # 2. get the mapping from the keys to fingers 
    def get_response_finger_map(self):
        """
        get the response mapping for the task 
        Creates a dictoinary 
        """
        # 2.1 load in the finger names corresponding to keys
        self.response_fingers = consts.response_fingers

        # 2.2 create a dictionary that maps response keys to fingers
        zip_iterator = zip(self.response_keys, self.response_fingers)
        return dict(zip_iterator)
    # 3. display the instructions for the task
    def display_instructions(self):
        """
        displays the instruction for the task
        Most tasks have the same instructions. (Tasks that have True/False responses)
        Those tasks that have different instructions will have their own routine   
        """
        # 3.1 create the instruction text
        # first use get_response_fingerMap to get the mapping between keys and finger names
        ## a dictionary called self.response_fingerMap is created!
        self.response_fingerMap = self.get_response_finger_map()
        hand = self.target_file['hand'][0]
        trueStr  = f"press {self.key_hand_dict[hand]['True'][0]} with {self.response_fingerMap[self.key_hand_dict[hand]['True'][0]]}"
        falseStr = f"press {self.key_hand_dict[hand]['False'][0]} with {self.response_fingerMap[self.key_hand_dict[hand]['False'][0]]}" 
 
        self.instruction_text = f"{self.task_name} task\n\nUse your {hand} hand\n\nIf true, {trueStr}\nIf false, {falseStr}"

        # 3.2 display the instruction text        
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        # instr.size = 0.8
        instr_visual.draw()
        self.window.flip()
    
    # 4. show task stimuli
    # After displaying the instructions, the stimuli will be shown. 
    # As the stimuli for most of the tasks are different, the show_stim routing was added
    # as an attribute of each specific task 

    # 5. get the trial response  
    def get_trial_response(self, wait_time, start_time, start_time_rt, **kwargs):
        """ 
        get the trial response. the ttl flag determines the timing 
        Args:
            wait_time     - how long will it wait for a response to be made
            start_time    - start time of the trial
            start_time_rt - ?????????
        *** if a task requires a different routine for getting the trial response, it will be added 
        as an attribute for that specific task. This is a general form!
        """
        # 5.1 reset the flag that shows whether a response was made
        self.response_made = False
        # 5.2 reset the reaction time
        self.rt = 0
        # 5.3 reset the list that will contain the pressed keys
        self.pressed_keys = []

        if self.ttl_flag: # if the user has chosen to use the ttl pulse
            while (ttl.clock.getTime() - start_time <= wait_time):
                ttl.check()
                self.pressed_keys.extend(event.getKeys(keyList= consts.response_keys, timeStamped=self.clock))
                if self.pressed_keys and not self.response_made: # if at least one press is made
                    self.response_made = True
                    self.rt = ttl.clock.getTime() - start_time_rt
        else: # do not wait for ttl pulse (behavioral)
            while (self.clock.getTime() - start_time <= wait_time): # and not resp_made:
                # get the keys that are pressed and the time they were pressed:
                ## two options here:
                ### 1. you can just check for the keys that are specified in const.response_keys
                ### 2. don't look for any specific keys and record every key that is pressed.
                ## the current code doesn't look for any specific keys and records evey key press
                # pressed_keys.extend(event.getKeys(keyList=consts.response_keys, timeStamped=self.clock))
                self.pressed_keys.extend(event.getKeys(keyList=None, timeStamped=self.clock))
                if self.pressed_keys and not self.response_made: # if at least one press is made
                    self.response_made = True
                    self.rt = self.clock.getTime() - start_time_rt
    # 6. check the trial response
    def check_trial_response(self, wait_time, trial_index, start_time, start_time_rt, **kwargs):
        """
        Checks whether the response made in the trial is correct
        Args:
            wait_time     - length of time allowed for a response to be made
            trial_index   - index of the trial (as read from the target file). Will be used to get the correct response  
            start_time    - the start time from when a response should have been made
            start_time_rt - ????????
        Returns:
            response_event - a dictionary with fields that correspond to the response:
                             "corr_key"    : the correct key that should have been pressed,
                             "pressed_key" : the key that was pressed,
                             "resp_made"   : whether a response was made or not (True: a response was made),
                             "corr_resp"   : whether the response was correct,
                             "rt"          : the reaction time
        """
        # get the row that correspsonds to the current trial
        row = self.target_file.iloc[trial_index] # the row of target dataframe corresponding to the current trial
        # get the list of keys that are to be pressed
        #** making sure that the trial_type is converted to str and it's not boolean
        self.correct_key_list = self.key_hand_dict[row['hand']][str(row['trial_type'])]
        self.correct_response = False

        # 6.1 get the trial response
        self.get_trial_response(wait_time, start_time, start_time_rt)

        # 6.2 check the trial response
        if self.pressed_keys and self.response_made:
            # assuming pressed_keys is sorted by timestamp; is it?
            # determine correct response based on first key press only
            if self.pressed_keys[0][0] == self.correct_key_list[0]:
                self.correct_response = True 
            elif self.pressed_keys[0][0] != self.correct_key_list[0]:
                self.correct_response = False
    
        # 6.3 determine the key that was pressed
        # the pressed key will be recorded even if it's the wrong key
        if not self.pressed_keys:
            # then no key was pressed
            resp_key = None
        else:
            resp_key = self.pressed_keys[0][0]

        response_event = {
            "corr_key": self.correct_key_list[0],
            "pressed_key": resp_key,
            "resp_made": self.response_made,
            "corr_resp": self.correct_response,
            "rt": self.rt
        }
        return response_event
    # 7. display the feedback for the current trial
    def display_trial_feedback(self, correct_response):
        """
        display the feedback for the current trial (correct response or not?)  
        Args:
            correct_response - a boolean variable. True if the response was correct, False otherwise
        """
        if correct_response:
            feedback = os.path.join(consts.stim_dir, self.study_name ,'correct.png')
        elif not correct_response:
            feedback = os.path.join(consts.stim_dir, self.study_name, 'incorrect.png')

        # display feedback on screen
        feedback = visual.ImageStim(self.window, feedback, pos=(0, 0)) # pos=pos
        feedback.draw()
        self.window.flip()
    # 8. Update the trial response (append the response to the list)
    def update_trial_response(self):
        # add additional variables to dict
        self.trial_response.update({'real_start_time': self.real_start_time, 
                                    'ttl_counter': self.ttl_count, 
                                    'ttl_time': self.ttl_time})

        self.all_trial_response.append(self.trial_response)
    # 9. Get the feedback for the task (the type of feedback is different across tasks)
    def get_task_feedback(self, dataframe, feedback_type):
        """
        gets overall feedback of the task based on the feedback type
        Args: 
            dataframe(pandas df)       -   response dataframe
            feedback_type (str)        -   feedback type for the task
        Returns:
            feedback (dict)     -   a dictionary containing measured feedback 
        """

        if feedback_type == 'rt':
            fb = dataframe.query('corr_resp==True').groupby(['run_name', 'run_iter'])['rt'].agg('mean')

            unit_mult = 1000 # multiplied by the calculated measure
            unit_str  = 'ms' # string representing the unit measure
        
        elif feedback_type == 'acc':
            fb = dataframe.groupby(['run_name', 'run_iter'])['corr_resp'].agg('mean')

            unit_mult = 100 # multiplied by the calculated measure
            unit_str  = '%' # string representing the unit measure
        else: # for flexion extension task
            fb = pd.DataFrame()
            unit_str = ''
        # add other possible types of feedback here   

        fb_curr = None
        fb_prev = None

        if not fb.empty:
            fb_curr = int(round(fb[-1] * unit_mult))
            if len(fb)>1:
                # get rt of prev. run if it exists
                fb_prev = int(round(fb[-2] * unit_mult))

        feedback = {'curr': fb_curr, 'prev': fb_prev, 'measure': unit_str} 

        return feedback 
    # 10. Get task response dataframe
    def get_task_response(self, all_trial_response):
        """
        get the responses made for the task and convert it to a dataframe
        Args:
            all_trial_response  -   responses made for all the trials in the task
        Outputs:
            response_df     -   dataframe containing the responses made for the task
        """
        # df for current data
        response_df = pd.concat([self.target_file, pd.DataFrame.from_records(all_trial_response)], axis=1)
        return response_df
    
    ## get the current time in the trial
    def get_current_trial_time(self):
        """
        gets the current time in the trial. The ttl_flag determines the timings.  
        """
        # gets the current time based on ttl_flag
        if self.ttl_flag:
            t_current = ttl.clock.getTime()
        else:
            t_current = self.clock.getTime()

        return t_current
    ## get the real start time of the trial
    def get_real_start_time(self, t0):
        """  
        gets the real start time and the ttl time
        If the user has chosen not to use the ttl pulse. 
        ttl_time is set to 0.
        Args:
            t0 - ?????????
        """
        if self.ttl_flag:
            # self.real_start_time = ttl.clock.getTime() - t0
            # self.ttl_time = ttl.time - t0
            self.real_start_time = ttl.clock.getTime()
            self.ttl_time = t0 - ttl.time
            # self.ttl_time = ttl.time
            self.ttl_count = ttl.count
            print(f"ttl count {self.ttl_count}")
            print(f"ttl time {self.ttl_time}")

        else:
            # self.real_start_time = self.clock.getTime() - t0
            self.real_start_time = self.clock.getTime()
            self.ttl_time = 0
            self.ttl_count = 0
    ##????????????????????????????????????
    def get_time_before_disp(self):
        # start timer before display
        if self.ttl_flag:
            self.t2 = ttl.clock.getTime()
        else:
            self.t2 = self.clock.getTime()

    ### quits the screen 
    def screen_quit(self):
        keys = event.getKeys()
        for key in keys:
            if 'q' and 'esc' in key:
                self.window.close()
                core.quit()
    ### shows fixation
    def show_fixation(self, t0, delta_t):
        if self.ttl_flag: # wait for ttl pulse
            while ttl.clock.getTime()-t0 <= delta_t:
                ttl.check()
        else: # do not wait for ttl pulse
            while self.clock.getTime()-t0 <= delta_t:
                pass
    

class VisualSearch(Task): 
    # @property
    # def instruction_text(self):
    #     return response dataframe

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(VisualSearch, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'rt' # reaction
        self.name          = 'visual_search'
    
    def _get_stims(self):
        # load target and distractor stimuli
        self.stims = [consts.stim_dir/ self.study_name / self.task_name/ f"{d}.png" for d in self.orientations]
        
        path_to_display = glob.glob(os.path.join(consts.target_dir, self.study_name, self.task_name, f'*display_pos_*_{self.target_num}*'))
        self.tf_display = pd.read_csv(path_to_display[0])

    def _get_trial_info(self):
        # for this task, no extra fields of the target file are needed
        # but just to be consistent with all the other tasks, I'm including a _get_trial_info routine
        # which is just calling get_trial_info from the parent class
        super().get_trial_info(self.trial)

    def _show_stim(self):
        # loop over items and display
        for idx in self.tf_display[self.tf_display['trial']==self.trial].index:
            stim_file = [file for file in self.stims if str(self.tf_display["orientation"][idx]) in file.stem] 
            
            stim = visual.ImageStim(self.window, str(stim_file[0]), pos=(self.tf_display['xpos'][idx], self.tf_display['ypos'][idx]), units='deg', size=self.item_size_dva)
            stim.draw()
        self.window.flip()
    
    def run(self):

        self.orientations = list([90, 180, 270, 360]) # ORDER SHOULD NOT CHANGE
        self.item_size_dva = 1

        # loop over trials and collect data
        self.all_trial_response = []

        # get display
        self._get_stims()

        # loop over trials
        for self.trial in self.target_file.index: 

            # get trial info
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            self.show_fixation(self.t0, self.start_time - self.t0)

            
            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)
            
            # flush any keys in buffer
            event.clearEvents()

            # display distract (+ target if present)
            self._show_stim()

            # Start timer before display (get self.t2)
            self.get_time_before_disp()

            # collect responses and update 
            wait_time = self.trial_dur

            self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                            trial_index = self.trial, 
                                                            start_time = self.t0, 
                                                            start_time_rt = self.t2)

            self.update_trial_response()

            # show feedback or fixation cross
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)
            
            # 6. 
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class NBack(Task):
    # @property
    # def instruction_text(self):
    #     return response dataframe

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(NBack, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'rt' # reaction
        self.name          = 'n_back'

    def _get_trial_info(self):
        super().get_trial_info(self.trial)
        stim_path = consts.stim_dir / self.study_name / self.task_name / self.target_file['stim'][self.trial]
        self.stim = visual.ImageStim(self.window, str(stim_path))
    
    def _show_stim(self):
        self.stim.draw()
        self.window.flip()
    
    def run(self):

        # loop over trials
        self.all_trial_response = [] # collect data

        for self.trial in self.target_file.index: 
            
            # show image
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            self.show_fixation(self.t0, self.start_time- self.t0)

            
            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # flush any keys in buffer
            event.clearEvents()

            # display stimulus
            self._show_stim()
            
            # Start timer before display (get self.t2)
            self.get_time_before_disp()

            # collect responses
            wait_time = self.trial_dur
            self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                            trial_index = self.trial, 
                                                            start_time = self.t0, 
                                                            start_time_rt = self.t2)

            # update trial response
            self.update_trial_response()

            # display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class SocialPrediction(Task):
    # @property
    # def instruction_text(self):
    #     return "Social Prediction Task\n\nYou have the following options\n\nHandShake = 1\nHug = 2\nHighFive = 3\nKiss = 4\n\nGo as fast as you can while being accurate"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(SocialPrediction, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'acc' # reaction
        self.name          = 'social_prediction'

    def _get_trial_info(self):
        super().get_trial_info(self.trial)
        video_file = self.target_file['stim'][self.trial]
        self.path_to_video = os.path.join(consts.stim_dir, self.study_name, self.task_name, "modified_clips", video_file)       
   
    def _get_first_response(self):
        # display trial feedback
        response_made = [dict['resp_made'] for dict in self.trial_response_all if dict['resp_made']]
        correct_response = False
        if response_made:
            response_made = response_made[0]
            correct_response = [dict['corr_resp'] for dict in self.trial_response_all if dict['resp_made']][0]
        else:
            response_made = False

        return response_made, correct_response
    
    def _get_response_event(self, response_made):
        # save response event
        if response_made:
            # save the first dict when response was made
            response_event = [dict for dict in self.trial_response_all if dict['resp_made']][0]
        else:
            response_event = [dict for dict in self.trial_response_all][0]

        return response_event
    
    def _show_stim(self):
        mov = visual.MovieStim3(self.window, self.path_to_video, flipVert=False, flipHoriz=False, loop=False)

        # play movie
        frames = []
        self.trial_response_all = []
        image = []
        wait_time = self.trial_dur
        
        if self.ttl_flag: # if the user chooses to wait for the ttl pulse
            while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

                # get trial response
                self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                                trial_index = self.trial, 
                                                                start_time = self.t0, 
                                                                start_time_rt = self.t2)
        else: 
            while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

                # get trial response
                self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                                trial_index = self.trial, 
                                                                start_time = self.t0, 
                                                                start_time_rt = self.t2)  
               
    def run(self):

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get trial_info + stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            self.show_fixation(self.t0, self.start_time - self.t0)

           # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # flush any keys in buffer
            event.clearEvents()

            # Start timer before display (get self.t2)
            self.get_time_before_disp()

            # display stims. The responses will be recorded and checked once the video is shown
            self._show_stim()

            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()
            
            # update response
            self.update_trial_response()

            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class SemanticPrediction(Task):
    # @property
    # def instruction_text(self):
    #     return "Language Prediction Task\n\nYou will read a sentence and decide if the final word of the sentence makes sense\n\nIf the word makes sense, press 3\n\nIf the word does not make sense, press 4\n\nAnswer as quickly and as accurately as possible"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(SemanticPrediction, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'rt' # reaction
        self.name          = 'semantic_prediction'

    def _get_trial_info(self):
        # get trial info from the target file
        super().get_trial_info(self.trial)
        self.stem = self.target_file['stim'][self.trial]
        self.stem = self.stem.split()
        self.stem_word_dur = self.target_file['stem_word_dur'][self.trial]
        self.last_word = self.target_file['last_word'][self.trial]
        self.last_word_dur = self.target_file['last_word_dur'][self.trial]

    def _get_stims(self):
        # get stim (i.e. word)
        self.stem = self.target_file['stim'][self.trial]
        self.stem = self.stem.split()
        self.stem_word_dur = self.target_file['stem_word_dur'][self.trial]
        self.start_time = self.target_file['start_time'][self.trial]
        self.last_word = self.target_file['last_word'][self.trial]
        self.last_word_dur = self.target_file['last_word_dur'][self.trial]
        self.iti_dur = self.target_file['iti_dur'][self.trial]
    
    def _show_stem(self):
        # display stem words for fixed time
        for word in self.stem:   
            self.word_start = self.get_current_trial_time()                     
            stim = visual.TextStim(self.window, text=word, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
            stim.draw()
            self.window.flip()
            # core.wait(self.stem_word_dur)

            # each word will remain on the screen for a certain amount of time (self.stem_word_dur)
            if self.ttl_flag: # wait for ttl pulse
                while ttl.clock.getTime()-self.word_start <= self.stem_word_dur:
                    ttl.check()
            else: # do not wait for ttl pulse
                while self.clock.getTime()-self.word_start <= self.stem_word_dur:
                    pass

    def _show_last_word(self):
        # display last word for fixed time
        self.word_start = self.get_current_trial_time()
        stim = visual.TextStim(self.window, text=self.last_word, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.draw()
        self.window.flip()
    
    def _show_stims_all(self):
        # show stem sentence
        self._show_stem()

        # display iti before final word presentation
        self.screen.fixation_cross()
        # core.wait(self.iti_dur)
        tc = self.get_current_trial_time()
        self.show_fixation(tc, self.iti_dur)

        # flush keys if any have been pressed
        event.clearEvents()

        # display last word for fixed time
        self._show_stim()
        self.window.flip()
    
    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # 1. show stems
            self._show_stem() 

            # 2. display fixation for the duration of the delay
            ## 2.1 get the current time
            t_stem_end = self.get_current_trial_time()
            ## 2.2 get the delay duration
            self.screen.fixation_cross()
            self.show_fixation(t_stem_end, self.iti_dur) 

            # 3. display the last word and collect reponse
            ## 3.1 display prob
            self._show_last_word()

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            # 3.3collect response
            wait_time = self.target_file['trial_dur_correct'][self.trial]

            self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                            trial_index = self.trial, 
                                                            start_time = self.t0, 
                                                            start_time_rt = self.t2)
            # 3.4 update response
            self.update_trial_response()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()

            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class ActionObservation(Task):
    # @property
    # def instruction_text(self):
    #     return "Action Observation Task\n\nYou have to decide whether the soccer player scores a goal\n\nYou will get feedback on every trial\n\nPress TRUE for goal\n\nPress FALSE for miss"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(ActionObservation, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'acc' # reaction
        self.name          = 'action_observation' 

    def _get_trial_info(self):
        super().get_trial_info(self.trial)
        video_file = self.target_file['stim'][self.trial]
        self.path_to_video = os.path.join(consts.stim_dir, self.study_name, self.task_name, "modified_clips", video_file)
    
    def _show_stim(self):
        mov = visual.MovieStim3(self.window, self.path_to_video, flipVert=False, flipHoriz=False, loop=False)

        # play movie
        frames = []
        self.trial_response_all = []
        image = []
        wait_time = self.trial_dur
        
        if self.ttl_flag:
            while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()
                
                # get trial response
                self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                                trial_index = self.trial, 
                                                                start_time = self.t0, 
                                                                start_time_rt = self.t2)
        else:
            while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()
                
                # get trial response
                self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                                trial_index = self.trial, 
                                                                start_time = self.t0, 
                                                                start_time_rt = self.t2)
    
    def run(self):

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get trial info and stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # flush any keys in buffer
            event.clearEvents()

            # Start timer before display (get self.t2)
            self.get_time_before_disp()

            # display stims and get trial response
            self._show_stim()

            # show feedback or fixation cross
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            # update response
            self.update_trial_response()

            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class TheoryOfMind(Task):
    # @property
    # def instruction_text(self):
    #     return "Theory of Mind Task\n\nYou will read a story and decide if the answer to the question is True or False.\n\nIf the answer is True, press 3\n\nIf the answers is False, press 4\n\nAnswer as quickly and as accurately as possible"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(TheoryOfMind, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'acc' # reaction
        self.name          = 'theory_of_mind'
    
    def _get_trial_info(self):
        super().get_trial_info(self.trial)
        # get stim (i.e. story)
        self.story = self.target_file['story'][self.trial]
        self.story_dur = self.target_file['story_dur'][self.trial]

        self.question = self.target_file['question'][self.trial]
        self.question_dur = self.target_file['question_dur'][self.trial]
    
    def _get_stims(self):
        # get stim (i.e. story)
        self.story = self.target_file['story'][self.trial]
        self.story_dur = self.target_file['story_dur'][self.trial]

        self.question = self.target_file['question'][self.trial]
        self.question_dur = self.target_file['question_dur'][self.trial]

        self.iti_dur = self.target_file['iti_dur'][self.trial]
        self.trial_dur = self.target_file['trial_dur'][self.trial]
        self.start_time = self.target_file['start_time'][self.trial]     
    
    def _show_story(self):
        # display story for fixed time    
        self.story_start = self.get_current_trial_time()                  
        stim = visual.TextStim(self.window, text=self.story, alignHoriz='center', wrapWidth=20, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.text = stim.text  # per PsychoPy documentation, this should reduce timing delays in displaying text
        stim.draw()
        self.window.flip()
        
        # the story will remain on the screen for a certain amount of time (self.story_dur)
        if self.ttl_flag: # wait for ttl pulse
            while ttl.clock.getTime()-self.story_start <= self.story_dur:
                    ttl.check()
        else: # do not wait for ttl pulse
            while self.clock.getTime()-self.story_start <= self.story_dur:
                pass

    def _show_stim(self):
        # display question for fixed time                       
        stim = visual.TextStim(self.window, text=self.question, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.text = stim.text  # per PsychoPy documentation, this should reduce timing delays in displaying text
        stim.draw()
        self.window.flip()
    
    def _show_stims_all(self):
        # show story
        self._show_story()

        # display iti before question presentation
        self.screen.fixation_cross()
        core.wait(self.iti_dur)

        # flush keys if any have been pressed
        event.clearEvents()

        # display question for fixed time
        self._show_stim()
        self.window.flip()
    
    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # 1. show story
            self._show_story()

            # 2. display fixation for the duration of the iti
            ## 2.1 get the current time
            t_story_end = self.get_current_trial_time()
            ## 2.2 get the iti duration
            self.screen.fixation_cross()
            self.show_fixation(t_story_end, self.iti_dur)
            ## 2.3 clear any button presses before collecting response
            event.clearEvents()

            # 3. display the probe and collect reponse
            ## 3.1 display prob
            self._show_stim()

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            ## 3.3 collect response
            wait_time = self.question_dur

            self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                            trial_index = self.trial, 
                                                            start_time = self.get_current_trial_time(), 
                                                            start_time_rt = self.t2)
            ## 3.4 update response
            self.update_trial_response()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class FingerSequence(Task):
    """
    a sequence of digits are shown to the participant.
    The participant needs to finish the sequence once!
    As the participant press the digits, an immediate feedback is shown:
        the digit turns green if the correct key was pressed
        the digit turns red if the incorrect key was pressed
    """

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(FingerSequence, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'acc' # reaction
        self.name          = 'finger_sequence'
        self.key_digit     = {
            'right':{'h':'1', 'j':'2', 'k':'3', 'l':'4'},
            'left' :{'a':'1', 's':'2', 'd':'3', 'f':'4'}
        }

    def _get_trial_info(self):
        """
        get the string(text) representing the fingers that are to be pressed from the target file
        in the target file, the field called sequence must contain a string with spaces between the keys
        """
        super().get_trial_info(self.trial)
        self.sequence_text = str(self.target_file['sequence'][self.trial])
        # self.announce_time = self.target_file['announce_time'][self.trial]
        
        # create a list of digits that are to be pressed
        self.digits_seq = self.sequence_text.split(" ")
    
    def display_instructions(self): # overriding the display instruction from the parent class
        # first use get_response_fingerMap to get the mapping between keys and finger names
        ## a dictionary called self.response_fingerMap is created!
        hand = self.target_file['hand'][0]
        self.response_fingerMap = self.get_response_finger_map()

        mapStr   = [f"press {item} with {self.response_fingerMap[item]}\n" for item in self.key_hand_dict[hand]['None']]
        temp_str = ''.join(mapStr)

        self.instruction_text = f"{self.task_name} task\n\nUse your {hand} hand:\n" + temp_str

        # display the instruction text        
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        # instr.size = 0.8
        instr_visual.draw()
        self.window.flip()

    def _show_sequence(self):
        """
        displays the sequence text
        creates a list containing text objects for each digit in the sequence
        """
        # the height option specifies the font size of the text that will be displayed
        self.seq_text_obj = []
        i_digit = 0 # digit counter
        x_pos = -5 # starting x position for the digit
        for d in self.digits_seq:
            self.seq_text_obj.append(visual.TextStim(self.window, text = d, color = [-1, -1, -1], height = 2, pos = [x_pos, 0]))
            self.seq_text_obj[i_digit].draw()
            i_digit = i_digit + 1
            x_pos = x_pos + 2

        self.window.flip()
    
    def _get_press_digit(self, press):
        """
        mapps the pressed key to the corresponding digit
        Args:
            press(str)   -   pressed key
        Returns:
            digit_press(str)    -   digit corresponding to the pressed key
        """
        # get the mapping from keys to digits
        map_k2d = self.key_digit[self.hand]

        # map the pressed keys to pressed digits
        if press in map_k2d:
            digit_press = map_k2d[press]
        else:
            digit_press = None

        return digit_press

    def _digit_feedback_color(self):
        # change the color of digits on the screen (as a form of immediate feedback)
        for obj in self.seq_text_obj:
            obj.draw()
        self.window.flip()
    
    def _wait_press(self):
        # waits for presses and once a press is made, check whether it's the correct key or not!
        # if correct, the digit turns into green, if incorrect, the digit turns into red
        
        ## each time a key is pressed, event.getKeys return a list
        ## the returned list has one element which is also a list ([[key, time]])
        ## the first index gives the key and the second index gives the time of press
        # each word will remain on the screen for a certain amount of time (self.stem_word_dur)
        if self.ttl_flag: # wait for ttl pulse
            ttl.check()
        else: # do not wait for ttl pulse
            pass
        press = event.getKeys(self.response_keys, timeStamped=self.clock) # records the pressed key
        if len(press)>0: # a press has been made`
            self.pressed_digits.append(self._get_press_digit(press[0][0])) # the pressed key is converted to its corresponding digit and appended to the list
            self.pressed_keys.append(press[0][0]) # get the pressed key
            self.press_times.append(press[0][1])  # get the time of press for the key

            try:
                if self.digits_seq[self.number_press] == self.pressed_digits[self.number_press]: # the press is correct
                    self.number_correct = self.number_correct + 1
                    self.seq_text_obj[self.number_press].setColor([-1, 1, -1]) # set the color of the corresponding digit to green
                    self._digit_feedback_color() # calls the function that sets the "immediate feedback color" of the digit
                else: # the press is incorrect
                    self.seq_text_obj[self.number_press].setColor([1, -1, -1]) # set the color of the corresponding digit to red
                    self._digit_feedback_color()
            except IndexError: # if the number of presses exceeds the length of the threshold
                self.correct_response = False
            finally:
                self.number_press = self.number_press + 1 # a press has been made => increase the number of presses
    
    def _get_trial_response(self, wait_time, trial_index, start_time, start_time_rt):
        # get the trial response and checks if the responses were correct
        # this task is different from the most tasks in that the participant needs
        # to make multiple responses! 
        row = self.target_file.iloc[trial_index] # the row of target dataframe corresponding to the current trial
        # get the list of keys that are to be pressed
        #** making sure that the trial_type is converted to str and it's not boolean
        self.correct_key_list = self.key_hand_dict[row['hand']][str(row['trial_type'])]
        self.correct_response = False
        self.response_made = False
        self.correct_response = False
        self.rt = 0

        self.number_press = 0 # number of presses made!
        self.number_correct = 0 # number of correct presses
        self.pressed_keys = [] # array containing pressed keys
        self.press_times = [] # array contaiing press times
        self.pressed_digits = [] # array containing pressed digits

        if self.ttl_flag:
            while (ttl.clock.getTime() - start_time <= wait_time): # and not resp_made:
                # it records key presses during this time window
                self._wait_press()
        else:
            while (self.clock.getTime() - start_time <= wait_time): # and not resp_made:
                self._wait_press()
                

        if self.pressed_keys and not self.response_made:
            self.response_made = True
            self.rt = self.press_times[0]
        else:
            self.response_made = False
            self.rt = None
            
        # if the number of presses made are correct and no error was made, the trial is counted as correct
        if (self.number_press == len(self.digits_seq)) and (self.number_correct == len(self.digits_seq)):
            # self.correct_trial +=1
            self.correct_response = True
        elif self.number_press > len(self.digits_seq):
            # self.error_trial +=1
            self.correct_response = False

        response_event = {
            "corr_digit": self.digits_seq,
            "resp_digit": self.pressed_digits,
            "resp_made": self.response_made,
            "corr_resp": self.correct_response,
            "rt": self.rt
            }

        return response_event

    def run(self):

        # loop over trials
        self.all_trial_response = [] # collect data

        for self.trial in self.target_file.index: 
            
            # show image
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # flush any keys in buffer
            event.clearEvents()

            # show the sequence of digits
            self._show_sequence()
            
            # Start timer before display (get self.t2)
            self.get_time_before_disp()

            # 2.collect responses and draw green rectangle (as a go signal)
            wait_time = self.trial_dur
            self.trial_response = self._get_trial_response(wait_time = wait_time,
                                                           trial_index = self.trial, 
                                                           start_time = self.t0 , 
                                                           start_time_rt = self.t2)

            # update trial response
            self.update_trial_response()

            # 3. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            # show the fixation cross for the duration of iti
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class SternbergOrder(Task):
    # This is a toy example with a toy target file. The target file might change in the future!
    """
    a list of digits (with length of 6) is shown sequentially (in a serial order)
    then a period of delay
    then prob. The prob will be something like 1<5. This is a True False response and means:
        Does 1 comes before 5 in the set?
        The participant needs to a) figure out whether 1 and 5 were in the set and 
                                 b) whether the order shown is correct

    The order of events in trial:
    1. show fixation (iti_dur)
    2. show digits serially
    3. show fixation for iti_dur seconds
    4. show prob digit and at the same time listen for response
    5. show fixation (iti_dur)
    """

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(SternbergOrder, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'acc' # reaction
        self.name          = 'sternberg_order'
    
    def _get_trial_info(self, trial_index):
        # get trial info from the target file
        super().get_trial_info(self.trial)
        self.stim      = self.target_file['stim'][self.trial]
        self.digits    = self.stim.split()
        self.digit_dur = self.target_file['digit_dur'][self.trial] # digit will stay on the screen for digit_dur sec
        self.delay_dur = self.target_file['delay_dur'][self.trial] # a delay period between memory set and probe set
        self.prob      = self.target_file['prob_stim'][self.trial] # stimuli that will be shown during the probe (this might change in the target file)
        self.prob_dur  = self.target_file['prob_dur'][self.trial] # probe will stay on the screen for prob_dur sec

    def _show_digits(self):
        # display digit for fixed time (self.digit_dur)
        for digit in self.digits:   
            self.digit_start = self.get_current_trial_time()                     
            stim = visual.TextStim(self.window, text=digit, pos=(0.0,0.0), color=(-1,-1,-1), units='deg', height = 1.5)
            stim.draw()
            self.window.flip()
            # core.wait(self.stem_word_dur)

            # each word will remain on the screen for a certain amount of time (self.stem_word_dur)
            if self.ttl_flag: # wait for ttl pulse
                while ttl.clock.getTime()-self.digit_start <= self.digit_dur:
                    ttl.check()
            else: # do not wait for ttl pulse
                while self.clock.getTime()-self.digit_start <= self.digit_dur:
                    pass

    def _show_prob(self):
        # display the prob on the screen (the probe comes after a delay period)
        self.prob_start = self.get_current_trial_time()
        # get the prob digits
        self.prob_dig = self.prob.split(" ")

        # the first digit of the prob
        dig_first = visual.TextStim(self.window, text=self.prob_dig[0], pos=(-1,0.0), color=(-1,-1,-1), units='deg', height = 1.5)
        
        # an arrow to show order   
        arrowVert = [(-1.6,0.2),(-1.6,-0.2),(-.8,-0.2),(-.8,-0.4),(0,0),(-.8,0.4),(-.8,0.2)]        # arrow = ShapeStim(self.window, vertices=arrowVert, fillColor='black', size=.5, lineColor='black')
        # arrowVert = [(0, 0), (1, 0), (1, 0.5), (1.5, 0), (1, -0.5), (1, 0)]
        # arrowVert = [(-1, 0), (0, 0), (0, 0.5), (1, 0), (0, -0.5), (0, 0)]
        arrow = ShapeStim(self.window, vertices=arrowVert, closeShape=True, lineWidth=3, pos=(0,0), ori=90, units = "deg", fillColor = [-1, -1, -1], lineColor = [-1, -1, -1])
        arrow.pos = [(-1, 1)]
        
        # the second digit of the prob
        dig_second = visual.TextStim(self.window, text=self.prob_dig[1], pos=(1,0.0), color=(-1,-1,-1), units='deg', height = 1.5)
        
        # draw the prob
        dig_first.draw()
        arrow.draw()
        dig_second.draw()
        
        self.window.flip()

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info(self.trial)

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # 1. show digits
            self._show_digits()

            # 2. display fixation for the duration of the delay
            ## 2.1 get the current time
            t_digit_end = self.get_current_trial_time()
            ## 2.2 get the delay duration
            self.screen.fixation_cross()
            self.show_fixation(t_digit_end, self.delay_dur)

            # 3. display the probe and collect reponse
            ## 3.1 display prob
            self._show_prob()

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            ## 3.3 collect response
            wait_time = self.prob_dur

            self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                            trial_index = self.trial, 
                                                            start_time = self.get_current_trial_time(), 
                                                            start_time_rt = self.t2)
            ## 3.4 update response
            self.update_trial_response()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6. 
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class VisuospatialOrder(Task):

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(VisuospatialOrder, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'acc' # reaction
        self.name          = 'visuospatial_order'

        # the coordinates are stored in lists and saved in the csv file
        # after reading the csv file, the coordinates lists will be loaded as strings, not lists
        # literal eval gets these columns and convert them into a "proper" type (in this case, list)
        self.target_file[['xys_stim', 'xys_prob']]= self.target_file[['xys_stim', 'xys_prob']].applymap(literal_eval)

    def _get_trial_info(self):

        super().get_trial_info(self.trial)
        self.trial_type             = self.target_file['trial_type'][self.trial]
        self.delay_dur              = self.target_file['delay_dur'][self.trial]
        self.dot_dur                = self.target_file['dot_dur'][self.trial]
        self.prob_dur               = self.target_file['prob_dur'][self.trial]
        self.circle_radius          = self.target_file['circle_radius'][self.trial]
        self.xys_stim               = self.target_file['xys_stim'][self.trial]
        self.xys_prob               = self.target_file['xys_prob'][self.trial]
        # self.angle_prob             = self.target_file['angle_prob'][self.trial]
    
    def _show_stim(self):
        # display dot for a fixed duration (self.dot_dur)
        for dot_idx in self.xys_stim: 
            # display a circle 
            # circle = visual.Circle(win=self.window, units='deg', radius=self.circle_radius, fillColor=[0, 0, 0], 
            #                        lineColor=[1, 1, 1], edges = 128, lineWidth = 5)  
            # circle.draw()
            # donutVert = [ [(-self.circle_radius-0.2,-self.circle_radius-0.2),(-self.circle_radius-0.2,self.circle_radius+0.2),(self.circle_radius+0.2,self.circle_radius+0.2),(self.circle_radius+0.2,-self.circle_radius-0.2)],
            #               [(-self.circle_radius,-self.circle_radius),(-self.circle_radius,self.circle_radius),(self.circle_radius,self.circle_radius),(self.circle_radius,-self.circle_radius)]]
            # donut = ShapeStim(win=self.window, units = 'deg', vertices=donutVert, fillColor=[0, 0, 0], lineColor=[1, 1, 1], lineWidth=1, size=.75, pos=(0,0))
            
            # donut.draw()

            rect = visual.Rect(win=self.window, units='deg', width = self.circle_radius+1, height = self.circle_radius+1, fillColor=[0, 0, 0], 
                                   lineColor=[1, 1, 1], lineWidth = 5)  
            rect.draw()
            
            dot_stim = visual.ElementArrayStim( win=self.window, units='deg', nElements=1, elementTex=None, elementMask="circle",
                                                xys=[dot_idx],
                                                sizes=1, colors = [-1, -1, -1])

            dot_stim = visual.Circle(win=self.window, units='deg', radius=0.3, fillColor=[-1,-1,-1], 
                                   lineColor=[-1, -1, -1], edges = 128, lineWidth = 5, pos = dot_idx)  
            dot_stim.draw()

            dot_stim.draw()
            self.window.flip()
            self.dot_start = self.get_current_trial_time()                     
            # each word will remain on the screen for a certain amount of time (self.stem_word_dur)
            if self.ttl_flag: # wait for ttl pulse
                while ttl.clock.getTime()-self.dot_start <= self.dot_dur:
                    ttl.check()
            else: # do not wait for ttl pulse
                while self.clock.getTime()-self.dot_start <= self.dot_dur:
                    pass

    def _show_prob(self):
        # display the prob on the screen (the probe comes after a delay period)
        self.prob_start = self.get_current_trial_time()

        # display a circle 
        # circle = visual.Circle(win=self.window, units='deg', radius=self.circle_radius, fillColor=[0, 0, 0], 
        #                         lineColor=[1, 1, 1], edges = 128, lineWidth = 5)  
        # circle.draw()
        rect = visual.Rect(win=self.window, units='deg', width = self.circle_radius+1, height = self.circle_radius+1, fillColor=[0, 0, 0], 
                                   lineColor=[1, 1, 1], lineWidth = 5)  
        rect.draw()

        dot_first = visual.Circle(win=self.window, units='deg', radius=0.3, fillColor=[-1,-1,-1], 
                                   lineColor=[-1, -1, -1], edges = 128, lineWidth = 5, pos = self.xys_prob[0]) 
        
        
        arrowVert = [(-1.6,0.2),(-1.6,-0.2),(-.8,-0.2),(-.8,-0.4),(0,0),(-.8,0.4),(-.8,0.2)]        # arrow = ShapeStim(self.window, vertices=arrowVert, fillColor='black', size=.5, lineColor='black')
        arrow = ShapeStim(self.window, vertices=arrowVert, closeShape=True, lineWidth=3, pos=(0,0), ori=90, units = "deg", fillColor = [-1, -1, -1], lineColor = [-1, -1, -1])
        arrow.pos = [self.xys_prob[0][0], self.xys_prob[0][1]+0.2]

        dot_second = visual.Circle(win=self.window, units='deg', radius=0.3, fillColor=[-1,-1,-1], 
                                   lineColor=[-1, -1, -1], edges = 128, lineWidth = 5, pos = self.xys_prob[1]) 
        
        # draw the prob
        dot_first.draw()
        arrow.draw()
        dot_second.draw()
        
        self.window.flip()

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # 1. show digits
            self._show_stim()

            # 2. display fixation for the duration of the delay
            ## 2.1 get the current time
            t_stim_end = self.get_current_trial_time()
            ## 2.2 get the delay duration
            self.screen.fixation_cross()
            self.show_fixation(t_stim_end, self.delay_dur)

            # 3. display the probe and collect reponse
            ## 3.1 display prob
            self._show_prob()

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            ## 3.3 collect response
            wait_time = self.prob_dur

            self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                            trial_index = self.trial, 
                                                            start_time = self.get_current_trial_time(), 
                                                            start_time_rt = self.t2)
            ## 3.4 update response
            self.update_trial_response()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class VisuospatialOrderV2(Task):

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(VisuospatialOrderV2, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'acc' # reaction
        self.name          = 'visuospatial_order'

        # the coordinates are stored in lists and saved in the csv file
        # after reading the csv file, the coordinates lists will be loaded as strings, not lists
        # literal eval gets these columns and convert them into a "proper" type (in this case, list)
        self.target_file[['xys_stim', 'xys_prob']]= self.target_file[['xys_stim', 'xys_prob']].applymap(literal_eval)

    def _get_trial_info(self):

        super().get_trial_info(self.trial)

        self.trial_type             = self.target_file['trial_type'][self.trial]
        self.delay_dur              = self.target_file['delay_dur'][self.trial]
        self.dot_dur                = self.target_file['dot_dur'][self.trial]
        self.prob_dur               = self.target_file['prob_dur'][self.trial]
        self.circle_radius          = self.target_file['circle_radius'][self.trial]
        self.xys_stim               = self.target_file['xys_stim'][self.trial]
        self.xys_prob               = self.target_file['xys_prob'][self.trial]
    
    def _fit_spline(self):
        # get the x and ys of the trial separated
        self.x = np.array([xys[0] for xys in self.xys_stim])
        self.y = np.array([xys[1] for xys in self.xys_stim])

        # append the starting x,y coordinates, closing the loop
        self.x = np.r_[self.x, self.x[0]]
        self.y = np.r_[self.y, self.y[0]]

        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        tck, u = interpolate.splprep([self.x, self.y], s=0, per=True)

        # evaluate the spline fits for 1000 evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

        return xi, yi

    def _show_stim(self):
        # fit a closed b spline curve to the x y coordinates of the dots
        self.xi, self.yi = self._fit_spline()

        for idx in range(len(self.xi)): 
            for f in range(1):

                dot_stim = visual.Circle(win=self.window, units='deg', radius=0.3, fillColor=[-1,-1,-1], 
                                    lineColor=[-1, -1, -1], edges = 128, lineWidth = 5, pos = [self.xi[idx], self.yi[idx]])  
                dot_stim.draw()

                dot_stim.draw()
                self.window.flip()
                # self.dot_start = self.get_current_trial_time()                     
            # # each word will remain on the screen for a certain amount of time (self.stem_word_dur)
            # if self.ttl_flag: # wait for ttl pulse
            #     while ttl.clock.getTime()-self.dot_start <= 0.001:
            #         ttl.check()
            # else: # do not wait for ttl pulse
            #     print(self.clock.getTime()-self.dot_start)
            #     while self.clock.getTime()-self.dot_start <= 0.001:
            #         pass

    def _show_prob(self):
        # display the prob on the screen (the probe comes after a delay period)
        self.prob_start = self.get_current_trial_time()

        dot_first = visual.Circle(win=self.window, units='deg', radius=0.3, fillColor=[-1,-1,-1], 
                                   lineColor=[-1, -1, -1], edges = 128, lineWidth = 5, pos = self.xys_prob[0]) 
        
        
        arrowVert = [(-1.6,0.2),(-1.6,-0.2),(-.8,-0.2),(-.8,-0.4),(0,0),(-.8,0.4),(-.8,0.2)]        # arrow = ShapeStim(self.window, vertices=arrowVert, fillColor='black', size=.5, lineColor='black')
        arrow = ShapeStim(self.window, vertices=arrowVert, closeShape=True, lineWidth=3, pos=(0,0), ori=90, units = "deg", fillColor = [-1, -1, -1], lineColor = [-1, -1, -1])
        arrow.pos = [self.xys_prob[0][0], self.xys_prob[0][1]+0.2]

        dot_second = visual.Circle(win=self.window, units='deg', radius=0.3, fillColor=[-1,-1,-1], 
                                   lineColor=[-1, -1, -1], edges = 128, lineWidth = 5, pos = self.xys_prob[1]) 
        
        # draw the prob
        dot_first.draw()
        arrow.draw()
        dot_second.draw()
        
        self.window.flip()

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # 1. show digits
            self._show_stim()

            # 2. display fixation for the duration of the delay
            ## 2.1 get the current time
            t_stim_end = self.get_current_trial_time()
            ## 2.2 get the delay duration
            self.screen.fixation_cross()
            self.show_fixation(t_stim_end, self.delay_dur)

            # 3. display the probe and collect reponse
            ## 3.1 display prob
            self._show_prob()

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            ## 3.3 collect response
            wait_time = self.prob_dur

            self.trial_response = self.check_trial_response(wait_time = wait_time, 
                                                            trial_index = self.trial, 
                                                            start_time = self.get_current_trial_time(), 
                                                            start_time_rt = self.t2)
            ## 3.4 update response
            self.update_trial_response()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6. 
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class FlexionExtension(Task):
    """
    flexion extension of toes! No particular feedback
    """
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(FlexionExtension, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'None' # reaction
        self.name          = 'flexion_extension'

    def _get_trial_info(self):
        # reads info from the target file
        super().get_trial_info(self.trial)
        self.stim = self.target_file['stim'][self.trial]
        self.stim_dur = self.target_file['stim_dur'][self.trial]
        self.trial_type = self.target_file['trial_type'][self.trial]
        self.display_trial_feedback = self.target_file['display_trial_feedback'][self.trial]
        # self.foot = self.target_file['foot'][self.trial]

    def display_instructions(self): # overriding the display instruction from the parent class
        self.instruction_text = f"{self.task_name} task \n\n flex and extend your right and left toes"   
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip() 
    
    def _show_stim(self):
        # displays the instruction:
        # either flexion or extension appears
        self.stim_pair = self.stim.split()
        n_rep = int(self.trial_dur/(self.stim_dur*2)) # number of times a stimulus pair will be repeated

        self.stim_act = np.tile(self.stim_pair, n_rep)


        for act in self.stim_act:
            self.act_start = self.get_current_trial_time()  
            # stim = visual.TextStim(self.window, text = self.foot + "\n "+ act, pos=(0.0,0.0), color=(-1,-1,-1), units='deg', height = 1.5)
            stim = visual.TextStim(self.window, text = act, pos=(0.0,0.0), color=(-1,-1,-1), units='deg', height = 1.5)
            stim.draw()
            self.window.flip()
            
            # core.wait(self.stem_word_dur)

            # each word will remain on the screen for a certain amount of time (self.stem_word_dur)
            if self.ttl_flag: # wait for ttl pulse
                while ttl.clock.getTime()-self.act_start <= self.stim_dur:
                    ttl.check()
            else: # do not wait for ttl pulse
                while self.clock.getTime()-self.act_start <= self.stim_dur:
                    pass       

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # 1. show actions
            self._show_stim()

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # if self.ttl_flag:
            #     while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
            #         ttl.check()
            # else:
            #     while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
            #         pass

            # update trial response
            self.trial_response = {}
            self.update_trial_response()

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class VerbGeneration(Task):
    # @property
    # def instruction_text(self):
    #     return "Verb Generation Task\n\nYou will read a series of nouns. For some nouns you will be asked to silently generate a verb.\n\nAnswer as quickly and as accurately as possible"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(VerbGeneration, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'None' # no feedback
        self.name          = 'verb_generation'
    
    def _get_trial_info(self):
        # get stim (i.e. story)
        super().get_trial_info(self.trial)
        self.noun = self.target_file['word'][self.trial]
        self.noun = self.noun.strip()
        self.start_time = self.target_file['start_time'][self.trial]
        
    def display_instructions(self): # overriding the display instruction from the parent class

        self.instruction_text = f"{self.task_name} task \n\n Silently read the words presented.  \n\n When GENERATE is shown, silently think of verbs that go with the words."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def _show_stim(self):
        # display word for fixed time    
        self.word_start = self.get_current_trial_time()                  
        stim = visual.TextStim(self.window, text=self.noun, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.text = stim.text  # per PsychoPy documentation, this should reduce timing delays in displaying text
        stim.draw()
        self.window.flip()
        
        # the word will remain on the screen for a certain amount of time (self.trial_dur)
        if self.ttl_flag: # wait for ttl pulse
            while ttl.clock.getTime()-self.word_start <= self.trial_dur - 0.1:
                    ttl.check()
        else: # do not wait for ttl pulse
            while self.clock.getTime()-self.word_start <= self.trial_dur - 0.1:
                pass

    def _show_instruct_read(self):
        # display instruction for fixed time
        self.instruct_read_start = self.get_current_trial_time()                        
        stim = visual.TextStim(self.window, text='READ', pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.text = stim.text  # per PsychoPy documentation, this should reduce timing delays in displaying text
        stim.draw()
        self.window.flip()

        # the instruction will remain on the screen for a certain amount of time (self.instruct_dur_secs)
        if self.ttl_flag: # wait for ttl pulse
            while ttl.clock.getTime()-self.instruct_read_start <= self.trial_dur - 0.1:
                    ttl.check()
        else: # do not wait for ttl pulse
            while self.clock.getTime()-self.instruct_read_start <= self.trial_dur - 0.1:
                pass

    def _show_instruct_generate(self):
        # display instruction for fixed time
        self.instruct_generate_start = self.get_current_trial_time()                       
        stim = visual.TextStim(self.window, text='GENERATE', pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.text = stim.text  # per PsychoPy documentation, this should reduce timing delays in displaying text
        stim.draw()
        self.window.flip()

        # the instruction will remain on the screen for a certain amount of time (self.instruct_dur_secs)
        if self.ttl_flag: # wait for ttl pulse
            while ttl.clock.getTime()-self.instruct_generate_start <= self.trial_dur - 0.1:
                    ttl.check()
        else: # do not wait for ttl pulse
            while self.clock.getTime()-self.instruct_generate_start <= self.trial_dur - 0.1:
                pass

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # 0.1 Show "GENERATE" instruction if this is half-way point
            if self.trial == int(self.target_file.index.stop/2):
                self._show_instruct_generate()

            # 1. show actions
            self._show_stim()

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class RomanceMovie(Task):
    # @property
    # def instruction_text(self):
    #     return "Romance Movie Task\n\nYou will passively watch a 30-second clip from a movie.  Please keep your head as still as possible."
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(RomanceMovie, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'None' # no feedback
        self.name          = 'romance_movie'

    def _get_trial_info(self):
        video_file = self.target_file['stim'][self.trial]
        self.iti_dur = self.target_file['iti_dur'][self.trial]
        self.trial_dur = self.target_file['trial_dur'][self.trial]
        self.start_time = self.target_file['start_time'][self.trial]
        self.path_to_video = os.path.join(consts.stim_dir, self.study_name, self.task_name, 'clips', video_file)
    
    def display_instructions(self): # overriding the display instruction from the parent class
        self.instruction_text = f"{self.task_name} task \n\n Please keep your head still while watching the movie clip."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip() 
    
    def _show_stim(self):
        mov = visual.MovieStim3(self.window, self.path_to_video, flipVert=False, flipHoriz=False, loop=False)

        # play movie
        frames = []
        self.trial_response_all = []
        image = []
        wait_time = self.trial_dur
        
        if self.ttl_flag: # if the user chooses to wait for the ttl pulse
            while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

        else: 
            while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            # 1. show actions
            self._show_stim()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class ActionObservationKnots(Task):
    # @property
    # def instruction_text(self):
    #     return "Action Observation Task\n\nYou will passively watch two 15-second clips.  Please keep your head as still as possible."
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(ActionObservationKnots, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'None' # no feedback
        self.name          = 'act_obs_knots'

    def _get_trial_info(self):
        video_file_action = self.target_file['stim_action'][self.trial]
        video_file_control = self.target_file['stim_control'][self.trial]
        self.iti_dur = self.target_file['iti_dur'][self.trial]
        self.trial_dur = self.target_file['trial_dur'][self.trial]
        self.start_time = self.target_file['start_time'][self.trial]
        self.path_to_video_action = os.path.join(consts.stim_dir, self.study_name, self.task_name, 'clips', video_file_action)
        self.path_to_video_control = os.path.join(consts.stim_dir, self.study_name, self.task_name, 'clips', video_file_control)
    
    def display_instructions(self): # overriding the display instruction from the parent class

        self.instruction_text = f"{self.task_name} task \n\n Keep your head still while watching to two clips. \n\n Try and remember the knot shown."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip() 
    
    def _show_stim_action(self):
        mov = visual.MovieStim3(self.window, self.path_to_video_action, flipVert=False, flipHoriz=False, loop=False)

        # play movie
        frames = []
        self.trial_response_all = []
        image = []
        wait_time = self.trial_dur
        
        if self.ttl_flag: # if the user chooses to wait for the ttl pulse
            while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

        else: 
            while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

    def _show_stim_control(self):
        mov = visual.MovieStim3(self.window, self.path_to_video_control, flipVert=False, flipHoriz=False, loop=False)

        # play movie
        frames = []
        self.trial_response_all = []
        image = []
        wait_time = self.trial_dur + self.trial_dur
        
        if self.ttl_flag: # if the user chooses to wait for the ttl pulse
            while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

        else: 
            while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                while mov.status != visual.FINISHED:
                    
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime 
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            # 1. show actions
            self._show_stim_action()

            # 2. display fixation for the duration of the iti
            ## 2.1 get the current time
            t_action_end = self.get_current_trial_time()
            ## 2.2 get the iti duration
            self.screen.fixation_cross()
            self.show_fixation(t_action_end, self.iti_dur)

            # 3. display the probe and collect reponse
            ## 3.2 get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            ## 3.1 display control video
            self._show_stim_control()

            # 4. display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()
            
            # 5 show fixation for the duration of the iti
            ## 5.1 get current time
            t_start_iti = self.get_current_trial_time()
            self.show_fixation(t_start_iti, self.iti_dur)

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

class Rest(Task):

    # @property

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num, ttl_flag):
        super(Rest, self).__init__(screen, target_file, run_end, task_name, study_name, target_num, ttl_flag)
        self.feedback_type = 'none' # reaction
        self.name          = 'rest'
    
    def display_instructions(self): # overriding the display instruction routine from the parent
        self.instruction_text = None
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        # instr.size = 0.8
        instr_visual.draw()
        self.window.flip()
    
    def _show_stim(self):
        # show fixation cross
        self.screen.fixation_cross()

    def run(self):
        # get current time (self.t0)
        self.t0 = self.get_current_trial_time()

        # loop over trials
        self.all_trial_response = [] # collect data

        for self.trial in self.target_file.index: 

            # show the fixation for the duration of iti
            self.show_fixation(self.t0, self.target_file['start_time'][self.trial])

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            # show stim
            self._show_stim()

            # Start timer before display (get self.t2)
            self.get_time_before_disp()

            # leave fixation on screen for `trial_dur`
            wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur'][self.trial]
            
            if self.ttl_flag:
                while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                    ttl.check()
            else:
                while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                    pass

            # update trial response
            self.trial_response = {}
            self.update_trial_response()

            # 6.
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf

TASK_MAP = {
    "visual_search": VisualSearch, # task_num 1
    "theory_of_mind": TheoryOfMind, # task_num 2
    "n_back": NBack, # task_num 3
    "social_prediction": SocialPrediction, # task_num 4
    "semantic_prediction": SemanticPrediction, # task_num 5
    "action_observation": ActionObservation, # task_num 6 
    "finger_sequence": FingerSequence, # task_num 7
    "sternberg_order": SternbergOrder, # task_num 8
    "visuospatial_order": VisuospatialOrder, # task 9
    "flexion_extension": FlexionExtension, # task_num 10
    "verb_generation": VerbGeneration, # task_num 11
    "romance_movie": RomanceMovie, #task_num 12
    "action_observation_knots": ActionObservationKnots, #task_num 13
    "rest": Rest, # task_num?
    }