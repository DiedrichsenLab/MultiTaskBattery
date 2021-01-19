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

import experiment_code.constants as consts
from experiment_code.screen import Screen
from experiment_code.ttl import ttl
import experiment_code.constants as const

class Task:
    """
    Task: takes in inputs from run_experiment.py and methods (e.g. 'instruction_text', 'save_to_df' etc) 
    are universal across all tasks.
    Each of other classes runs a unique task given input from target files and from the Task class
    (VisualSearch, SemanticPrediction, NBack, SocialPrediction, ActionObservation).
    """

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        self.screen = screen
        self.window = screen.window
        self.monitor = screen.monitor
        self.target_file = target_file
        self.run_end = run_end
        self.clock = core.Clock()
        self.study_name = study_name
        self.task_name = task_name
        self.target_num = target_num

    @property
    def instruction_text(self):
        # return None
        hand = self.target_file['hand'][0]
        return f"{self.task_name} task\n\nUse your {hand} hand\n\nIf true, press {consts.key_hand_dict[hand][True][0]} with {consts.key_hand_dict[hand][True][1]}\nIf false, press {consts.key_hand_dict[hand][False][0]} with {consts.key_hand_dict[hand][False][1]}"
    
    def get_resp_df(self, all_trial_response):
        """
        get the responses made for the task and convert it to a dataframe
        Args:
            all_trial_response  -   responses made for all the trials in the task
        Outputs:
            resp_df     -   dataframe containing the responses made for the task
        """
        # df for current data
        resp_df = pd.concat([self.target_file, pd.DataFrame.from_records(all_trial_response)], axis=1)
        return resp_df
    
    def run(self, df):
        return df

    def display_instructions(self):
        instr = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        # instr.size = 0.8
        instr.draw()
        self.window.flip()

    def get_correct_key(self, trial_index):
        row = self.target_file.iloc[trial_index] # the row of target dataframecorresponding to the current trial 
        return row['hand'], row['trial_type']
        # return consts.key_hand_dict[row['hand']][row['trial_type']][0]

    def get_feedback(self, dataframe, feedback_type):
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

    def display_feedback(self, feedback_text):
        feedback = visual.TextStim(self.window, text=feedback_text, color=[-1, -1, -1])
        feedback.draw()
        self.window.flip()

    def display_end_run(self):
        # end_exper_text = f"End of run {self.run_num}\n\nTake a break!"
        end_exper_text = f"End of run\n\nTake a break!"
        end_experiment = visual.TextStim(self.window, text=end_exper_text, color=[-1, -1, -1])
        end_experiment.draw()
        self.window.flip()

    def display_trial_feedback(self, correct_response):
        if correct_response:
            feedback = os.path.join(consts.stim_dir, self.study_name ,'correct.png')
        elif not correct_response:
            feedback = os.path.join(consts.stim_dir, self.study_name, 'incorrect.png')

        # display feedback on screen
        feedback = visual.ImageStim(self.window, feedback, pos=(0, 0)) # pos=pos
        feedback.draw()
        self.window.flip()
    
    def get_trial_response(self, wait_time, trial_index, start_time, start_time_rt, **kwargs):

        #----------------------------------------------------------------------------------------
        # the number of responses made will be different for motor sequence task.
        # in the motor sequence task multiple presses will be made.
        # the code should be written to incorporate those changes
        # For most tasks, one press is enough. Even if more than one press is made, 
        # the first press will be considered!
        # For motor sequence task more than one press must be made and each press made 
        # will be compared with the corresponding key
        ##  Each task has an attribute called number_key_press
        ##  At the end of each trial, collect the number of presses made in the trial
        ##  How many presses should have been made?
        ##  Compare each press made with the correct corresponding key
        #----------------------------------------------------------------------------------------
        self.correct_key = 0
        self.hand, self.trial_type = self.get_correct_key(trial_index)

        self.correct_key = self.get_correct_key(trial_index)
        self.response_made = False
        self.correct_response = False
        self.rt = 0
        pressed_keys = []
        
        while (self.clock.getTime() - start_time <= wait_time): # and not resp_made:
            pressed_keys.extend(event.getKeys(consts.response_keys, timeStamped=self.clock))
            # print(pressed_keys.extend(event.getKeys(consts.response_keys, timeStamped=self.clock)))
            if pressed_keys and not self.response_made:
                self.response_made = True
                self.rt = self.clock.getTime() - start_time_rt
                # assuming pressed_keys is sorted by timestamp; is it?
                # determine correct response based on first key press only
                if pressed_keys[0][0] == self.correct_key:
                    self.correct_response = True 
                elif pressed_keys[0][0] != self.correct_key:
                    self.correct_response = False

        # determine the key that was pressed
        # the pressed key will be recorded even if the wrong key was pressed
        if not pressed_keys:
            # then no key was pressed
            resp_key = None
        else:
            resp_key = pressed_keys[0][0]


        response_event = {
            "corr_key": self.correct_key,
            "pressed_key": resp_key,
            # "key_presses": pressed_keys,
            "resp_made": self.response_made,
            "corr_resp": self.correct_response,
            "rt": self.rt
        }
        return response_event

    def _show_stim(self):
        raise NotImplementedError
   
    def update_trial_response(self):
        # add additional variables to dict
        self.trial_response.update({'real_start_time': self.real_start_time})

        self.all_trial_response.append(self.trial_response)
    
    def screen_quit(self):
        keys = event.getKeys()
        for key in keys:
            if 'q' and 'esc' in key:
                self.window.close()
                core.quit()


class VisualSearch(Task): 
    # @property
    # def instruction_text(self):
    #     return response dataframe

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(VisualSearch, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'rt' # reaction
        self.name          = 'visual_search'
        self.response_keys = ['d', 'f', 'j', 'k']

        # assign keys to hands
        self.key_hand_dict = {
            'right': {    # right hand
                True:  [self.response_keys[2], 'Index'], # index finger
                False: [self.response_keys[3], 'Middle'],  # middle finger
                },
            'left': {   # left hand
                False:[self.response_keys[0], 'Middle'], # index finger
                True: [self.response_keys[1], 'Index'],  # middle finger
                },
            } 
    
    def _get_stims(self):
        # load target and distractor stimuli
        self.stims = [consts.stim_dir/ self.study_name / self.task_name/ f"{d}.png" for d in self.orientations]
        
        path_to_display = glob.glob(os.path.join(consts.target_dir, self.study_name, self.task_name, f'*display_pos_*_{self.target_num}*'))
        self.tf_display = pd.read_csv(path_to_display[0])

    def _show_stim(self):
        # loop over items and display
        for idx in self.tf_display[self.tf_display['trial']==self.trial].index:
            stim_file = [file for file in self.stims if str(self.tf_display["orientation"][idx]) in file.stem] 
            
            stim = visual.ImageStim(self.window, str(stim_file[0]), pos=(self.tf_display['xpos'][idx], self.tf_display['ypos'][idx]), units='deg', size=self.item_size_dva)
            stim.draw()
    
    def run(self):

        # get current time
        t0 = self.clock.getTime()

        self.orientations = list([90, 180, 270, 360]) # ORDER SHOULD NOT CHANGE
        self.item_size_dva = 1

        # loop over trials and collect data
        self.all_trial_response = []

        # get display
        self._get_stims()

        # loop over trials
        for self.trial in self.target_file.index: 

            # before image is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-t0 <= self.target_file['start_time'][self.trial]:
                pass

            # flush any keys in buffer
            event.clearEvents()

            # display distract (+ target if present)
            self._show_stim()
            self.window.flip()

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - t0

            # Start timer before display
            t2 = self.clock.getTime()

            # collect responses and update 
            wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur'][self.trial]
            self.trial_response = self.get_trial_response(wait_time = wait_time,
                                                          trial_index = self.trial, 
                                                          start_time = t0, 
                                                          start_time_rt = t2)

            self.update_trial_response()

            # show feedback or fixation cross
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf

class NBack(Task):
    # @property
    # def instruction_text(self):
    #     return response dataframe

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(NBack, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'rt' # reaction
        self.name          = 'n_back'
        self.response_keys = ['d', 'f', 'j', 'k']

        # assign keys to hands
        self.key_hand_dict = {
            'right': {    # right hand
                True:  [self.response_keys[2], 'Index'], # index finger
                False: [self.response_keys[3], 'Middle'],  # middle finger
                },
            'left': {   # left hand
                False:[self.response_keys[0], 'Middle'], # index finger
                True: [self.response_keys[1], 'Index'],  # middle finger
                },
            } 

    def _get_stims(self):
        # show image
        stim_path = consts.stim_dir / self.study_name / self.task_name / self.target_file['stim'][self.trial]
        self.stim = visual.ImageStim(self.window, str(stim_path))
    
    def _show_stim(self):
        self.stim.draw()
    
    def run(self):

        # get current time
        t0 = self.clock.getTime()

        # loop over trials
        self.all_trial_response = [] # collect data

        for self.trial in self.target_file.index: 
            
            # show image
            self._get_stims()

             # before image is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-t0 <= self.target_file['start_time'][self.trial]:
                pass

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - t0

            # flush any keys in buffer
            event.clearEvents()

            # display stimulus
            self._show_stim()
            self.window.flip()

            # Start timer before display
            t2 = self.clock.getTime()

            # collect responses
            wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur'][self.trial]
            self.trial_response = self.get_trial_response(wait_time = wait_time,
                                    trial_index = self.trial, 
                                    start_time = t0, 
                                    start_time_rt = t2)

            # update trial response
            self.update_trial_response()

            # display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            # option to quit screen
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf

class SocialPrediction(Task):
    # @property
    # def instruction_text(self):
    #     return "Social Prediction Task\n\nYou have the following options\n\nHandShake = 1\nHug = 2\nHighFive = 3\nKiss = 4\n\nGo as fast as you can while being accurate"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(SocialPrediction, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'acc' # reaction
        self.name          = 'social_prediction'
        self.response_keys = ['d', 'f', 'j', 'k']

        # assign keys to hands
        self.key_hand_dict = {
            'right': {    # right hand
                True:  [self.response_keys[2], 'Index'], # index finger
                False: [self.response_keys[3], 'Middle'],  # middle finger
                },
            'left': {   # left hand
                False:[self.response_keys[0], 'Middle'], # index finger
                True: [self.response_keys[1], 'Index'],  # middle finger
                },
            } 

    def _get_stims(self):
        video_file = self.target_file['stim'][self.trial]
        self.path_to_video = os.path.join(consts.stim_dir, self.study_name, self.task_name, "modified_clips", video_file)

    def _get_trial_response(self):
        self.correct_key = 0
        self.correct_key = self.get_correct_key(self.trial)
        self.response_made = False
        self.correct_response = False
        self.rt = 0
        pressed_keys = []
        
        pressed_keys.extend(event.getKeys(consts.response_keys, timeStamped=self.clock))
        if pressed_keys and not self.response_made:
            self.response_made = True
            self.rt = self.clock.getTime() - self.t2
            # assuming pressed_keys is sorted by timestamp; is it?
            # determine correct response based on first key press only
            if pressed_keys[0][0] == self.correct_key:
                self.correct_response = True 
            elif pressed_keys[0][0] != self.correct_key:
                self.correct_response = False

        response_event = {
            "corr_key": self.correct_key,
            # "key_presses": pressed_keys,
            "resp_made": self.response_made,
            "corr_resp": self.correct_response,
            "rt": self.rt
            }

        return response_event
   
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
        wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur'][self.trial]
        while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
        
            # play movie
            while mov.status != visual.FINISHED:
                
                # draw frame to screen
                mov.draw()
                self.window.flip()

                # get trial response
                self.trial_response_all.append(self._get_trial_response()) 
            
            # check for response again after movie has finished playing
            self.trial_response_all.append(self._get_trial_response())
    
    def run(self):

        # get current time
        self.t0 = self.clock.getTime()

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_stims()

            # before word is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-self.t0 <= self.target_file['start_time'][self.trial]:
                pass

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - self.t0

            # flush any keys in buffer
            event.clearEvents()

            # Start timer before display
            self.t2 = self.clock.getTime()

            # display stims
            self._show_stim()

            # show feedback or fixation cross
            response_made, correct_response = self._get_first_response()
            if self.target_file['display_trial_feedback'][self.trial] and response_made:
                self.display_trial_feedback(correct_response = correct_response)
            else:
                self.screen.fixation_cross()

            # get response event
            self.trial_response = self._get_response_event(response_made = response_made)

            # update response
            self.update_trial_response()

            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf

class SemanticPrediction(Task):
    # @property
    # def instruction_text(self):
    #     return "Language Prediction Task\n\nYou will read a sentence and decide if the final word of the sentence makes sense\n\nIf the word makes sense, press 3\n\nIf the word does not make sense, press 4\n\nAnswer as quickly and as accurately as possible"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(SemanticPrediction, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'rt' # reaction
        self.name          = 'semantic_prediction'
        self.response_keys = ['d', 'f', 'j', 'k']

        # assign keys to hands
        self.key_hand_dict = {
            'right': {    # right hand
                True:  [self.response_keys[2], 'Index'], # index finger
                False: [self.response_keys[3], 'Middle'],  # middle finger
                },
            'left': {   # left hand
                False:[self.response_keys[0], 'Middle'], # index finger
                True: [self.response_keys[1], 'Index'],  # middle finger
                },
            } 
    
    def _get_stims(self):
        # get stim (i.e. word)
        self.stem = self.target_file['stim'][self.trial]
        self.stem = self.stem.split()
        self.stem_word_dur = self.target_file['stem_word_dur'][self.trial]

        self.last_word = self.target_file['last_word'][self.trial]
        self.last_word_dur = self.target_file['last_word_dur'][self.trial]

        self.iti_dur = self.target_file['iti_dur'][self.trial]
    
    def _show_stem(self):
        # display stem words for fixed time
        for word in self.stem:                         
            stim = visual.TextStim(self.window, text=word, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
            stim.draw()
            self.window.flip()
            core.wait(self.stem_word_dur)

    def _show_stim(self):
        # display last word for fixed time
        stim = visual.TextStim(self.window, text=self.last_word, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.draw()
    
    def _show_stims_all(self):
        # show stem sentence
        self._show_stem()

        # display iti before final word presentation
        self.screen.fixation_cross()
        core.wait(self.iti_dur)

        # flush keys if any have been pressed
        event.clearEvents()

        # display last word for fixed time
        self._show_stim()
        self.window.flip()
    
    def run(self):

        # get current time
        t0 = self.clock.getTime()

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_stims()

            # before word is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-t0 <= self.target_file['start_time'][self.trial]:
                pass

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - t0

            # display stem
            self._show_stims_all() 

            # Start timer before display
            t2 = self.clock.getTime()

            # collect response
            wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur_correct'][self.trial]
            self.trial_response = self.get_trial_response(wait_time = wait_time,
                                    trial_index = self.trial, 
                                    start_time = t0, 
                                    start_time_rt = t2)

           # update response
            self.update_trial_response()

            # display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()

            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf

class ActionObservation(Task):
    # @property
    # def instruction_text(self):
    #     return "Action Observation Task\n\nYou have to decide whether the soccer player scores a goal\n\nYou will get feedback on every trial\n\nPress TRUE for goal\n\nPress FALSE for miss"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(ActionObservation, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'acc' # reaction
        self.name          = 'action_observation'
        self.response_keys = ['d', 'f', 'j', 'k']

        # assign keys to hands
        self.key_hand_dict = {
            'right': {    # right hand
                True:  [self.response_keys[2], 'Index'], # index finger
                False: [self.response_keys[3], 'Middle'],  # middle finger
                },
            'left': {   # left hand
                False:[self.response_keys[0], 'Middle'], # index finger
                True: [self.response_keys[1], 'Index'],  # middle finger
                },
            } 

    def _get_stims(self):
        video_file = self.target_file['stim'][self.trial]
        self.path_to_video = os.path.join(consts.stim_dir, self.study_name, self.task_name, "modified_clips", video_file)

    def _get_trial_response(self):
        self.correct_key = 0
        self.correct_key = self.get_correct_key(self.trial)
        self.response_made = False
        self.correct_response = False
        self.rt = 0
        pressed_keys = []
        
        pressed_keys.extend(event.getKeys(consts.response_keys, timeStamped=self.clock))
        if pressed_keys and not self.response_made:
            self.response_made = True
            self.rt = self.clock.getTime() - self.t2
            # assuming pressed_keys is sorted by timestamp; is it?
            # determine correct response based on first key press only
            if pressed_keys[0][0] == self.correct_key:
                self.correct_response = True 
            elif pressed_keys[0][0] != self.correct_key:
                self.correct_response = False

        response_event = {
            "corr_key": self.correct_key,
            # "key_presses": pressed_keys,
            "resp_made": self.response_made,
            "corr_resp": self.correct_response,
            "rt": self.rt
            }

        return response_event
   
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
        wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur'][self.trial]
        while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
        
            # play movie
            while mov.status != visual.FINISHED:
                
                # draw frame to screen
                mov.draw()
                self.window.flip()

                # get trial response
                self.trial_response_all.append(self._get_trial_response()) 
            
            # check for response again after movie has finished playing
            self.trial_response_all.append(self._get_trial_response())
    
    def run(self):

        # get current time
        self.t0 = self.clock.getTime()

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_stims()

            # before word is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-self.t0 <= self.target_file['start_time'][self.trial]:
                pass

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - self.t0

            # flush any keys in buffer
            event.clearEvents()

            # Start timer before display
            self.t2 = self.clock.getTime()

            # display stims and get trial response
            self._show_stim()

            # show feedback or fixation cross
            response_made, correct_response = self._get_first_response()
            if self.target_file['display_trial_feedback'][self.trial] and response_made:
                self.display_trial_feedback(correct_response = correct_response)
            else:
                self.screen.fixation_cross()

            # get response event
            self.trial_response = self._get_response_event(response_made = response_made)

            # update response
            self.update_trial_response()

            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf

class TheoryOfMind(Task):
    # @property
    # def instruction_text(self):
    #     return "Theory of Mind Task\n\nYou will read a story and decide if the answer to the question is True or False.\n\nIf the answer is True, press 3\n\nIf the answers is False, press 4\n\nAnswer as quickly and as accurately as possible"
    
    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(TheoryOfMind, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'acc' # reaction
        self.name          = 'theory_of_mind'
        self.response_keys = ['d', 'f', 'j', 'k']

        # assign keys to hands
        self.key_hand_dict = {
            'right': {    # right hand
                True:  [self.response_keys[2], 'Index'], # index finger
                False: [self.response_keys[3], 'Middle'],  # middle finger
                },
            'left': {   # left hand
                False:[self.response_keys[0], 'Middle'], # index finger
                True: [self.response_keys[1], 'Index'],  # middle finger
                },
            } 
    
    def _get_stims(self):
        # get stim (i.e. story)
        self.story = self.target_file['story'][self.trial]
        self.story_dur = self.target_file['story_dur'][self.trial]

        self.question = self.target_file['question'][self.trial]
        self.question_dur = self.target_file['question_dur'][self.trial]

        self.iti_dur = self.target_file['iti_dur'][self.trial]
    
    def _show_story(self):
        # display story for fixed time                       
        stim = visual.TextStim(self.window, text=self.story, alignHoriz='center', pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.draw()
        self.window.flip()
        core.wait(self.story_dur)

    def _show_stim(self):
        # display question for fixed time                       
        stim = visual.TextStim(self.window, text=self.question, pos=(0.0,0.0), color=(-1,-1,-1), units='deg')
        stim.draw()
        self.window.flip()
        core.wait(self.question_dur)
    
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

        # get current time
        t0 = self.clock.getTime()

        # loop over trials
        self.all_trial_response = [] # pre-allocate 

        for self.trial in self.target_file.index: 

            # get stims
            self._get_stims()

            # before word is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-t0 <= self.target_file['start_time'][self.trial]:
                pass

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - t0

            # display stem
            self._show_stims_all() 

            # Start timer before display
            t2 = self.clock.getTime()

            # collect response
            wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur_correct'][self.trial]
            self.trial_response = self.get_trial_response(wait_time = wait_time,
                                    trial_index = self.trial, 
                                    start_time = t0, 
                                    start_time_rt = t2)

           # update response
            self.update_trial_response()

            # display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response) 
            else:
                self.screen.fixation_cross()

            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf

class FingerSequence(Task):

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(FingerSequence, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'acc' # reaction
        self.name          = 'finger_sequence'
        self.response_keys = ['s', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
        
        # assign keys to hands
        self.key_hand_dict = {
            'right':{
                True: [self.response_keys[4], 'Index'],  # Index
                True: [self.response_keys[5], 'Middle'], # Middle
                True: [self.response_keys[6], 'Ring'],   # Ring
                True: [self.response_keys[7], 'Pinky'],  # Pinky
            },
            'left':{
                True: [self.response_keys[0], 'Pinky'],  # Pinky
                True: [self.response_keys[1], 'Ring'],   # Ring
                True: [self.response_keys[2], 'Middle'], # Middle
                True: [self.response_keys[3], 'Index'],  # Index
            }
        }
        
    def _get_stims(self):
        """
        get the string(text) representing the fingers that are to be pressed from the target file
        in the target file, the field called sequence must contain a string with spaces between the keys
        """
        self.sequence_text = str(self.target_file['sequence'][self.trial])

    def _show_stim(self):
        """
        displays the sequence text
        """
        seq = visual.TextStim(self.window, text=self.sequence_text, color=[-1, -1, -1], height = 2)
        # instr.size = 0.8
        seq.draw()
        # self.window.flip()

        # return

    def run(self):

        # get current time
        t0 = self.clock.getTime()

        # loop over trials
        self.all_trial_response = [] # collect data

        for self.trial in self.target_file.index: 
            
            # show image
            self._get_stims()

            # get the keys for each press
            keys = self.sequence_text.split(" ") # this line assumes that in the target file spaces are included between each press

             # before image is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-t0 <= self.target_file['start_time'][self.trial]:
                pass

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - t0

            # flush any keys in buffer
            event.clearEvents()

            # display stimulus
            self._show_stim()
            self.window.flip()

            # Start timer before display
            t2 = self.clock.getTime()

            # collect responses
            wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur'][self.trial]
            self.trial_response = self.get_trial_response(wait_time = wait_time,
                                    trial_index = self.trial, 
                                    start_time = t0, 
                                    start_time_rt = t2)

            # update trial response
            self.update_trial_response()

            # display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            # option to quit screen
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf



class Rest(Task):

    # @property

    def __init__(self, screen, target_file, run_end, task_name, study_name, target_num):
        super(Rest, self).__init__(screen, target_file, run_end, task_name, study_name, target_num)
        self.feedback_type = 'none' # reaction
        self.name          = 'rest'

    def instruction_text(self):
        return None
    
    def _show_stim(self):
        # show fixation cross
        self.screen.fixation_cross()

    def run(self):
        # get current time
        t0 = self.clock.getTime()

        # loop over trials
        self.all_trial_response = [] # collect data

        for self.trial in self.target_file.index: 

             # before image is shown: fixation cross hangs on screen for iti_dur
            while self.clock.getTime()-t0 <= self.target_file['start_time'][self.trial]:
                pass

            # collect real_start_time for each block
            self.real_start_time = self.clock.getTime() - t0

            # show stim
            self._show_stim()

            # Start timer before display
            t2 = self.clock.getTime()

            # leave fixation on screen for `trial_dur`
            wait_time = self.target_file['start_time'][self.trial] + self.target_file['trial_dur'][self.trial]
            while (self.clock.getTime() - t0 <= wait_time): # and not resp_made:
                pass

            # update trial response
            self.trial_response = {}
            self.update_trial_response()

            # option to quit screen
            self.screen_quit()

        # get the response dataframe
        rDf = self.get_resp_df(all_trial_response=self.all_trial_response)

        return rDf


#TASK_MAP = {
#    "visual_search": VisualSearch,
#    "n_back": NBack,
#    "social_prediction": SocialPrediction,
#    "semantic_prediction": SemanticPrediction,
#    "action_observation": ActionObservation,
#    "theory_of_mind": TheoryOfMind,
#    "rest": Rest,
#}

TASK_MAP = {
    "visual_search": VisualSearch,
    "theory_of_mind": TheoryOfMind,
    "n_back": NBack,
    "social_prediction": SocialPrediction,
    "semantic_prediction": SemanticPrediction,
    "action_observation": ActionObservation,
    "rest": Rest,
    }

TASK_MAP = {
    "visual_search": VisualSearch,
    "theory_of_mind": TheoryOfMind,
    "n_back": NBack,
    "social_prediction": SocialPrediction,
    "semantic_prediction": SemanticPrediction,
    "action_observation": ActionObservation,
    "finger_sequence": FingerSequence,
    "rest": Rest,
    }