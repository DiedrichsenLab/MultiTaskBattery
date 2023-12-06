# Task Class defintions
# March 2021: First version: Ladan Shahshahani  - Maedbh King - Suzanne Witt,
# Revised 2023: Jorn Diedrichsen, Ince Hussain, Bassel Arafat

# import libraries
from pathlib import Path
import os
import pandas as pd
import numpy as np
import glob

from psychopy import visual, sound, core, event, constants, gui  # data, logging
from psychopy.visual import ShapeStim

import experiment_code.utils as ut
from experiment_code.screen import Screen
from experiment_code.ttl_clock import TTLClock

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

    def __init__(self, info, screen, ttl_clock, const):

        # Pointers to Screen and experimental constants
        self.screen  = screen
        self.window = screen.window # Shortcut to window
        self.const   = const
        self.ttl_clock       =  ttl_clock  # This is a reference to the clock of the run
        self.name        = info['task_name']
        self.code        = self.name[:4]
        self.target_file = info['target_file']

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file,sep='\t')

    def display_instructions(self):
        """
        displays the instruction for the task
        Most tasks have the same instructions. (Tasks that have True/False responses)
        Those tasks that have different instructions will have their own routine
        """
        true_str = f"if True press {self.const.response_keys[1]}"
        false_str = f"if False press {self.const.response_keys[2]}"

        self.instruction_text = f"{self.name} task\n\n {true_str} \n {false_str}"

        # 3.2 display the instruction text
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        # instr.size = 0.8
        instr_visual.draw()
        self.window.flip()

    def run(self):
        """Loop over trials and collects data
        Data will br stored in self.trial_data

        Returns:
            info (pd.DataFrame): _description_
        """
        self.trial_data = [] # an empty list which will be appended with the responses from each trial

        for i,trial in self.trial_info.iterrows():
            t_data = trial.copy()
            # Wait for the the start of next trial
            t_data['real_start_time'],t_data['start_ttl'],t_data['start_ttl_time'] = self.ttl_clock.wait_until(self.start_time + trial.start_time )
            # Run the trial
            t_data = self.run_trial(t_data)
            # Append the trial data
            self.trial_data.append(t_data)
        self.trial_data = pd.DataFrame(self.trial_data)


    def wait_response(self, start_time, max_wait_time):
        """
        waits for a response to be made and then returns the response
        Args:
            start_time (float): the time the RT-period started
            max_wait_time (float): How long to wait maximally
        Returns:
            key (str): the key that was pressed ('none' if no key was pressed)
            rt (float): the reaction time (nan if no key was pressed)
        """
        response_made = False
        key = 'none'
        rt = np.nan

        while (self.ttl_clock.get_time() - start_time <= max_wait_time) and not response_made:
            self.ttl_clock.update()
            keys=event.getKeys(keyList= self.const.response_keys, timeStamped=self.ttl_clock.clock)
            if len(keys)>0:
                response_made = True
                key = keys[0][0]
                rt = keys[0][1] - start_time
        return key, rt

    def display_trial_feedback(self, give_feedback,correct_response):
        """
        display the feedback for the current trial using the color of the fixation cross

        Args:
            give_feedback (bool):
                If true, gives informative feedback - otherwise just shows fixation cross
            correct_response (bool):
                Response was correct?, False otherwise
        """
        if give_feedback:
            if correct_response:
                self.screen.fixation_cross('green')
            else:
                self.screen.fixation_cross('red')
        else:
            self.screen.fixation_cross('white')

    def save_data(self, subj_id, run_num):
        """Saves the data to the trial data file

        Args:
            subj_id (str): Subject id to determine name
            run_num (int): Number of run - inserted as first column
        """
        self.trial_data.insert(0, 'run_num', [run_num]*len(self.trial_data))
        trial_data_file = self.const.data_dir / subj_id / f"{subj_id}_task-{self.code}.tsv"
        ut.append_data_to_file(trial_data_file, self.trial_data)

    # 8. Get the feedback for the task (the type of feedback is different across tasks)
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


    # save the response for the task
    def save_task_response(self, response_df, file_path):
        """
        gets the response dataframe and save it
        Args:
            response_df(pd dataframe) -   response dataframe
            file_path                 -   path where response will be saved

        """
        # check if a task response file already exists and load it and then update it
        if os.path.isfile(file_path):
            target_file_results = pd.read_csv(file_path)
            target_resp_df      = pd.concat([target_file_results, response_df], axis=0, sort=False)
        else: # if there is no existing data, just save current data
            target_resp_df = response_df
        # save all data
        target_resp_df.to_csv(file_path, index=None, header=True)

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

class NBack(Task):
    # def instruction_text(self):
    #     return response dataframe

    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'acc'

    def init_task(self):
        """Read the target file and get all the stimuli necessary"""
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file,sep='\t')
        self.stim=[]
        for stim in self.trial_info['stim']:
            stim_path = self.const.stim_dir / self.name / stim
            self.stim.append(visual.ImageStim(self.window, str(stim_path)))

    def run_trial(self,trial):
        """Runs a single trial of the nback task (after it started)
        Args:
            trial (pd.Series):
                Row of trial_info, with all information about the trial

        Returns:
            trial (pd.Series):
                Row of trial_data with all response data added
        """
        # Flush any keys in buffer
        event.clearEvents()

        # display stimulus
        self.stim[trial['trial_num']].draw()
        self.window.flip()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # check if response is correct
        trial['response'] = (key == self.const.response_keys[1])
        trial['correct'] = (trial['response'] == trial['trial_type'])

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])
        return trial
    
class VerbGeneration(Task):
    # def instruction_text(self):
    #     return "Verb Generation Task\n\nYou will read a series of nouns. For some nouns you will be asked to silently generate a verb.\n\nAnswer as quickly and as accurately as possible"

    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'None'

    def init_task(self):
        """ Initialize task-specific settings. """
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')
        self.trial_info['noun'] = self.trial_info['stim'].str.strip()

    def display_instructions(self): # overriding the display instruction from the parent class

        self.instruction_text = f"{self.name} task \n\n Silently read the words presented.  \n\n When GENERATE is shown, silently think of verbs that go with the words."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def show_stim(self, noun):
        """ Display a word for a fixed time. """
        stim = visual.TextStim(self.window, text=noun, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
        stim.draw()
        self.window.flip()

    def display_generate_instruction(self):
        """ Display the 'GENERATE' instruction. """
        generate_instr = visual.TextStim(self.window, text='GENERATE', pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
        generate_instr.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the VerbGeneration task. """
        # Wait for the start time of the trial
        real_start_time, start_ttl, start_ttl_time = self.ttl_clock.wait_until(trial['start_time'])

        # Display word
        self.show_stim(trial['noun'])

        # Optionally display GENERATE instruction at the halfway point
        if trial.name == int(self.trial_info.index.stop / 2):
            self.display_generate_instruction()

        # Here you can handle the response collection if needed

        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl
        trial['start_ttl_time'] = start_ttl_time

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= False, correct_response = False)
        return trial
    
class Rest(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'none'
        self.name          = 'rest'

    def display_instructions(self): # overriding the display instruction routine from the parent
        self.instruction_text = 'Rest: Fixate on the cross'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        # instr.size = 0.8
        instr_visual.draw()
        self.window.flip()

    def show_stim(self):
        # show fixation cross
        self.screen.fixation_cross()

    def run_trial(self,trial):
        # get current time (self.t0)
        self.screen.fixation_cross()
        self.ttl_clock.wait_until(self.start_time + trial['trial_dur'])
        return trial

class FlexionExtension(Task):
    """
    Flexion extension of toes! No particular feedback.
    """
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'None'
    
    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f"{self.name} task \n\n Flex and extend your right and left toes"   
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        real_start_time, start_ttl, start_ttl_time = self.ttl_clock.wait_until(trial['start_time'])
        # Handle the display of each action
        self.stim_pair = trial['stim'].split()
        n_rep = int(trial['trial_dur'] / (trial['stim_dur'] * 2))
        stim_actions = np.tile(self.stim_pair, n_rep)

        for action in stim_actions:
            # Display action
            stim = visual.TextStim(self.window, text=action, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=1.5)
            stim.draw()
            self.window.flip()

            # Wait for the duration of the stimulus
            start_time = self.ttl_clock.get_time()
            self.ttl_clock.wait_until(start_time + trial['stim_dur'])

        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl
        trial['start_ttl_time'] = start_ttl_time

        # No response is expected in this task, so return trial as is
        return trial
    
class TongueMovement(Task):
    """
    Tongue movement following Buckner et al., 2022! No particular feedback.
    """
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'None'
    
    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f"{self.name} task \n\n Move your tongue left to right touching your upper premolar teeth"   
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the tonguemovement task. """
        # Wait for the start time of the trial
        real_start_time, start_ttl, start_ttl_time = self.ttl_clock.wait_until(trial['start_time'])

        # draw fixation cross without flipping
        self.screen.fixation_cross(flip=False)
        
        # Check the trial_type and display the corresponding stimulus
        if trial['trial_type'] == 'right':
            # If trial_type is 'right', show the black circle around the fixation cross
            circle_visual = visual.Circle(self.window, radius=1, edges= 32, fillColor=None, lineColor='black')
            circle_visual.draw()

        self.window.flip()

        # Here you can handle the response collection if needed

        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl
        trial['start_ttl_time'] = start_ttl_time

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= False, correct_response = False)
        return trial
    
class AuditoryNarrative(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'None'  

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = 'Auditory Narrative Task\n\nListen to the narrative attentively.'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the AuditoryNarrative task. """

        self.screen.fixation_cross()
        # Wait for the start time of the trial
        real_start_time, start_ttl, start_ttl_time = self.ttl_clock.wait_until(trial['start_time'])

        # Load and play audio stimulus for the current trial
        audio_path = self.const.stim_dir / self.name / trial['stim']
        audio_stim = sound.Sound(str(audio_path))
        audio_stim.play()

        # Wait for the duration of the trial while audio is playing
        self.ttl_clock.wait_until(real_start_time + trial['trial_dur'])

        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl
        trial['start_ttl_time'] = start_ttl_time

        # Assuming no response is expected in this task
        return trial

class RomanceMovie(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.name = 'romance_movie' 

    def display_instructions(self):
        self.instruction_text = "In this task, you will watch short clips from a romance movie. Please keep your head still and pay attention to the screen."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        # Assuming that 'stim' column in trial contains the file name of the video clip
        movie_file_name = trial['stim']

        # Construct the movie file path
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / movie_file_name

        # Convert Path object to string for compatibility
        movie_path_str = str(movie_path)

        # Create a MovieStim3 object
        movie_clip = visual.MovieStim3(self.window, movie_path_str, loop=False)

        start_time = self.ttl_clock.get_time()
        while self.ttl_clock.get_time() - start_time < trial['trial_dur']:
            if movie_clip.status == visual.FINISHED:
                break
            movie_clip.draw()
            self.window.flip()
            self.ttl_clock.update()
        return trial
    
class SpatialNavigation(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'None' 

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        start_location = self.trial_info.iloc[0]['location_1']
        end_location = self.trial_info.iloc[0]['location_2']

        self.instruction_text = (f"Spatial navigation task\n\n"
                                    f"Imagine walking around your childhood home\n"
                                    f"Start in the {start_location} – end in the {end_location}\n"
                                    f"Focus on the fixation cross")
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1],  wrapWidth=400)
        instr_visual.draw()
        self.window.flip()

    
    def run_trial(self,trial):
        # Wait for the start time of the trial
        real_start_time, start_ttl, start_ttl_time = self.ttl_clock.wait_until(trial['start_time'])

        self.screen.fixation_cross()
        self.ttl_clock.wait_until(self.start_time + trial['trial_dur'])

        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl
        trial['start_ttl_time'] = start_ttl_time

        return trial
    
class TheoryOfMind(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'acc'

    def init_task(self):
        """ Initialize task - read the target information into the trial_info dataframe """
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        """ Display the instructions for the Theory of Mind task """
        instructions = ("Theory of Mind Task\n\nYou will read a story and decide if the answer to the question "
                        "is True or False.\n\nIf the answer is True, press {}\n\nIf the answer is False, press {}\n\n"
                        "Answer as quickly and as accurately as possible").format(self.const.response_keys[1], self.const.response_keys[2])

        instr_visual = visual.TextStim(self.window, text=instructions, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Runs a single trial of the Theory of Mind task """

        # Wait for the start time of the trial
        real_start_time, start_ttl, start_ttl_time = self.ttl_clock.wait_until(trial['start_time'])
        
        # Display story
        story_stim = visual.TextStim(self.window, text=trial['story'], alignHoriz='center', wrapWidth=20, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
        story_stim.draw()
        self.window.flip()
        _, _, _ = self.ttl_clock.wait_until(real_start_time + trial['story_dur'])


        # Display question
        question_stim = visual.TextStim(self.window, text=trial['question'], pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
        question_stim.draw()
        self.window.flip()


        # Collect response (after question display duration)
        trial['response_key'], trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])

        # Check if response is correct
        trial['response'] = trial['response_key'] in [self.const.response_keys[1], self.const.response_keys[2]]
        trial['correct'] = (trial['response'] == trial['condition'])

        # Provide feedback if necessary
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl
        trial['start_ttl_time'] = start_ttl_time

        return trial





### ====================================================================================================
# What follows is tasks we still need to modify
### ====================================================================================================

class VisualSearch(Task):
    # @property
    # def instruction_text(self):
    #     return response dataframe

    def __init__(self, info, screen, const):
        super().__init__(info, screen, const)
        self.feedback_type = 'acc'

    def init_task(self):
        # load target and distractor stimuli
        # self.stims = [consts.stim_dir/ self.study_name / self.name/ f"{d}.png" for d in self.orientations]
        super().init_task()
        self.stims = [self.const.stim_dir/ self.name/ f"{d}.png" for d in self.orientations]

        display_file = os.path.join(self.const.target_dir,  self.name, self.trial_info['display_file'])
        self.tf_display = pd.read_csv(display_file)

    def show_stim(self):
        # loop over items and display
        for idx in self.tf_display[self.tf_display['trial']==self.trial].index:
            stim_file = [file for file in self.stims if str(self.tf_display["orientation"][idx]) in file.stem]

            stim = visual.ImageStim(self.window, str(stim_file[0]), pos=(self.tf_display['xpos'][idx], self.tf_display['ypos'][idx]), units='deg', size=self.item_size_dva)
            stim.draw()
        self.screen.window.flip()

    def run(self):

        self.orientations = list([90, 180, 270, 360]) # ORDER SHOULD NOT CHANGE
        self.item_size_dva = 1

        # loop over trials and collect data
        self.all_trial_response = []

        # get display
        self._get_stims()

        # loop over trials
        for i,trial in self.trial_info.iterrows():

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

class SocialPrediction(Task):
    def instruction_text(self):
        return "Social Prediction Task\n\nYou have the following options\n\nHandShake = 1\nHug = 2\nHighFive = 3\nKiss = 4\n\nGo as fast as you can while being accurate"

    def __init__(self, info, screen, const):
        super().__init__(info, screen, const)
        self.feedback_type = 'acc'

    def init_task(self):
        super().init_task()
        # stim_path = consts.stim_dir / self.study_name / self.name / self.target_file['stim'][self.trial]
        self.stim=[]
        for stim in self.target_file['stim']:
            video_file = self.const.stim_dir / self.name / stim
            self.stim.append(visual.ImageStim(self.window, str(stim_path)))

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

        while mov.status != visual.FINISHED:
            if self.ttl_flag:
                while (ttl.clock.getTime() - self.t0 <= wait_time):
                    ttl.check()
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

            else:
                while (self.clock.getTime() - self.t0 <= wait_time):
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

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

    def __init__(self, info, screen, const):
        super().__init__(info, screen, const)
        self.feedback_type = 'acc'

    def init_task(self):
        super().init_task()
        # stim_path = consts.stim_dir / self.study_name / self.name / self.target_file['stim'][self.trial]
        for i,trial in self.trial_info.iterrows():
            pass

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

    def __init__(self, info, screen, const):
        super().__init__(info, screen, const)
        self.feedback_type = 'acc'

    def init_task(self):
        super().init_task()
        for i,trial in self.trial_info.iterrows():
            video_file = self.trial_info['stim'][self.trial]
        # self.path_to_video = os.path.join(consts.stim_dir, self.study_name, self.name, "modified_clips", video_file)
        self.path_to_video = os.path.join(consts.stim_dir, self.study_name, self.name, "modified_clips", video_file)

    def _show_stim(self):
        mov = visual.MovieStim3(self.window, self.path_to_video, flipVert=False, flipHoriz=False, loop=False)

        # play movie
        self.trial_response_all = []
        wait_time = self.trial_dur
        print(f"mov status : {mov.status}")

        # while mov.status != visual.FINISHED:
        while mov.status != constants.FINISHED:

            if self.ttl_flag:
                while (ttl.clock.getTime() - self.t0 <= wait_time):
                    ttl.check()
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

                    # get trial response
                    self.trial_response = self.check_trial_response(wait_time = wait_time,
                                                                    trial_index = self.trial,
                                                                    start_time = self.t0,
                                                                    start_time_rt = self.t2)

            else:
                while (self.clock.getTime() - self.t0 <= wait_time):
                    # draw frame to screen
                    mov.draw()
                    self.window.flip()

                    # get trial response
                    self.trial_response = self.check_trial_response(wait_time = wait_time,
                                                                    trial_index = self.trial,
                                                                    start_time = self.t0,
                                                                    start_time_rt = self.t2)

        # if self.ttl_flag:
        #     while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
        #         # play movie
        #         while mov.status != visual.FINISHED:
        #             ttl.check()
        #             # draw frame to screen
        #             mov.draw()
        #             self.window.flip()

        #         # get trial response
        #         self.trial_response = self.check_trial_response(wait_time = wait_time,
        #                                                         trial_index = self.trial,
        #                                                         start_time = self.t0,
        #                                                         start_time_rt = self.t2)
        # else:
        #     while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
        #         # play movie
        #         while mov.status != visual.FINISHED:

        #             # draw frame to screen
        #             mov.draw()
        #             self.window.flip()

        #         # get trial response
        #         self.trial_response = self.check_trial_response(wait_time = wait_time,
        #                                                         trial_index = self.trial,
        #                                                         start_time = self.t0,
        #                                                         start_time_rt = self.t2)

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

class FingerSequence(Task):
    """
    a sequence of digits are shown to the participant.
    The participant needs to finish the sequence once!
    As the participant press the digits, an immediate feedback is shown:
        the digit turns green if the correct key was pressed
        the digit turns red if the incorrect key was pressed
    """

    def __init__(self, screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save = True):
        super().__init__(screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save_response = save)
        self.feedback_type = 'acc' # reaction
        self.name          = 'finger_sequence'

        # create a dictionary to map keys to digits
        response_left  = consts.response_keys[0:4]
        response_right = consts.response_keys[4:8]
        map_right      = dict(zip(response_right, ('2', '3', '4', '5')))
        map_left       = dict(zip(response_left, ('2', '3', '4', '5')))
        self.key_digit = dict(zip(('right', 'left'), (map_right, map_left)))

    def _get_trial_info(self):
        """
        get the string(text) representing the fingers that are to be pressed from the target file
        in the target file, the field called sequence must contain a string with spaces between the keys
        """
        super().get_trial_info(self.trial)
        # print(f"trial_number {self.trial}")
        self.sequence_text = str(self.target_file['sequence'][self.trial])

        # create a list of digits that are to be pressed
        self.digits_seq = self.sequence_text.split(" ")

    def display_instructions(self): # overriding the display instruction from the parent class
        # first use get_response_fingerMap to get the mapping between keys and finger names
        ## a dictionary called self.response_fingerMap is created!
        hand = self.target_file['hand'][0]
        self.response_fingerMap = self.get_response_finger_map()
        # For 1 press the index finger ()


        mapStr   = [f"for {item} press {self.response_fingerMap[item]}\n" for item in self.key_hand_dict[hand]['None']]
        temp_str = ''.join(mapStr)

        self.instruction_text = f"{self.name} task\n\nUse your {hand} hand:\n" + temp_str

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
        else:
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

    def __init__(self, screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save = True):
        super().__init__(screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save_response = save)
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

    def __init__(self, screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save = True):
        super().__init__(screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save_response = save)
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

    def __init__(self, screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save = True):
        super().__init__(screen, target_file, run_end, name, task_num, study_name, target_num, ttl_flag, save_response = save)
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

class ActionObservationKnots(Task):
    # @property
    # def instruction_text(self):
    #     return "Action Observation Task\n\nYou will passively watch two 15-second clips.  Please keep your head as still as possible."

    def __init__(self, screen, target_file, run_end, task_name, task_num, study_name, target_num, ttl_flag, save = True):
        super().__init__(screen, target_file, run_end, task_name, task_num, study_name, target_num, ttl_flag, save_response = save)
        self.feedback_type = 'None' # no feedback
        self.name          = 'action_observation_knots'

    def _get_trial_info(self):
        video_file = self.target_file['stim'][self.trial]
        self.iti_dur = self.target_file['iti_dur'][self.trial]
        self.trial_dur = self.target_file['trial_dur'][self.trial]
        self.start_time = self.target_file['start_time'][self.trial]
        self.end_time = self.target_file['end_time'][self.trial]
        self.path_to_video = os.path.join(consts.stim_dir, self.task_name, 'clips', video_file)
        # self.mov = visual.MovieStim3(self.window, self.path_to_video, flipVert=False, flipHoriz=False, loop=False)

    def display_instructions(self): # overriding the display instruction from the parent class

        self.instruction_text = f"{self.task_name} task \n\n Keep your head still while watching the two clips. \n\n Try and remember the knot shown."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def _show_stim_movie(self, path_to_movie):

        mov = visual.MovieStim3(self.window, path_to_movie, flipVert=False, flipHoriz=False, loop=False)
        # play movie
        self.trial_response_all = []
        wait_time = self.trial_dur

        if self.ttl_flag: # if the user chooses to wait for the ttl pulse
            while (ttl.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                # while self.mov.status != constants.FINISHED:
                ttl.check()
                # draw frame to screen
                mov.draw()
                self.window.flip()

        else:
            while (self.clock.getTime() - self.t0 <= wait_time): # and not resp_made:
                # play movie
                # while self.mov.status != constants.FINISHED:
                # draw frame to screen
                mov.draw()
                self.window.flip()

        # mov.stop()

    def run(self):
        # run the task

        # loop over trials
        self.all_trial_response = [] # pre-allocate
        for self.trial in self.target_file.index:

            self.trial_response = {}

            # get stims
            self._get_trial_info()

            # get current time (self.t0)
            self.t0 = self.get_current_trial_time()

            # show the fixation for the duration of iti
            # wait here till the startTime
            self.show_fixation(self.t0, self.start_time - self.t0)

            # collect real_start_time for each block (self.real_start_time)
            self.get_real_start_time(self.t0)

            ## get the time before collecting responses (self.t2)
            self.get_time_before_disp()

            self._show_stim_movie(self.path_to_video)

            self.update_trial_response()
            # display trial feedback
            if self.target_file['display_trial_feedback'][self.trial] and self.response_made:
                self.display_trial_feedback(correct_response = self.correct_response)
            else:
                self.screen.fixation_cross()

            self.screen_quit()

        # get the response dataframe
        rDf = self.get_task_response(all_trial_response=self.all_trial_response)

        return rDf



