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

import MultiTaskBattery.utils as ut
from MultiTaskBattery.screen import Screen
from MultiTaskBattery.ttl_clock import TTLClock

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
        self.code        = info['code']
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

    def screen_quit(self):
        """ Checks for quit or escape key presses and quits the experiment if necessary """
        keys = event.getKeys()
        for key in keys:
            if 'q' and 'esc' in key:
                self.window.close()
                core.quit()

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

        # check if response is correct (jorn check plz, problematic for no feedback trials and for last trial if last task it can cause the run to be less than 30 secs)
        trial['response'] = (key == self.const.response_keys[1])
        trial['correct'] = (trial['response'] == trial['trial_type'])

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])
        return trial

class Rest(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'one'
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

class VerbGeneration(Task):
    # def instruction_text(self):
    #     return "Verb Generation Task\n\nYou will read a series of nouns. For some nouns you will be asked to silently generate a verb.\n\nAnswer as quickly and as accurately as possible"

    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

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

        # Display word
        self.show_stim(trial['noun'])

        # display GENERATE instruction at the halfway point (jorn check this plz, this replaces word 7 with generate)
        if trial.name == len(self.trial_info) // 2:
            self.display_generate_instruction()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])


        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)
        return trial

class TongueMovement(Task):
    """
    Tongue movement following Buckner et al., 2022! No particular feedback.
    """
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f"{self.name} task \n\n Move your tongue left to right touching your upper premolar teeth"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the tonguemovement task. """

        # draw fixation cross without flipping
        self.screen.fixation_cross(flip=False)

        # Check the trial_type and display the corresponding stimulus
        if trial['trial_type'] == 'right':
            # If trial_type is 'right', show the black circle around the fixation cross
            circle_visual = visual.Circle(self.window, radius=1, edges= 32, fillColor=None, lineColor='black')
            circle_visual.draw()

        self.window.flip()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class AuditoryNarrative(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f'{self.name} Task\n\nListen to the narrative attentively.'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the AuditoryNarrative task. """
        self.screen.fixation_cross()

        # Load and play audio stimulus for the current trial
        audio_path = self.const.stim_dir / self.name / trial['stim']
        audio_stim = sound.Sound(str(audio_path))
        audio_stim.play()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class RomanceMovie(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.name = 'romance_movie'

    def display_instructions(self):
        self.instruction_text = f"{self.name} Task\n\n You will watch short clips from a romance movie. Please keep your head still and pay attention to the screen."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        # Get the file name
        movie_file_name = trial['stim']

        # Construct the movie file path
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / movie_file_name

        # Convert Path object to string for compatibility
        movie_path_str = str(movie_path)

        # Create a MovieStim3 object
        movie_clip = visual.MovieStim3(self.window, movie_path_str, loop=False)

        movie_clip.draw()
        self.window.flip()

        while movie_clip.status != visual.FINISHED:
            movie_clip.draw()
            self.window.flip()

        # Display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class SpatialNavigation(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        start_location = self.trial_info.iloc[0]['location_1']
        end_location = self.trial_info.iloc[0]['location_2']

        self.instruction_text = (f"{self.name} Task \n\n"
                                    f"Imagine walking around your childhood home\n"
                                    f"Start in the {start_location} – end in the {end_location}\n"
                                    f"Focus on the fixation cross")
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1],  wrapWidth=400)
        instr_visual.draw()
        self.window.flip()

    def run_trial(self,trial):
        self.screen.fixation_cross()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

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
        instructions = (f"{self.name} Task \n\nYou will read a story and decide if the answer to the question "
                        "is True or False.\n\nIf the answer is True, press {}\n\nIf the answer is False, press {}\n\n"
                        ).format(self.const.response_fingers[1], self.const.response_fingers[2])

        instr_visual = visual.TextStim(self.window, text=instructions, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Runs a single trial of the Theory of Mind task """
        event.clearEvents()

        # Display story
        story_stim = visual.TextStim(self.window, text=trial['story'], alignHoriz='center', wrapWidth=20, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
        story_stim.draw()
        self.window.flip()

       # wait until story duration
        core.wait(trial['story_dur'])


        # Display question
        question_stim = visual.TextStim(self.window, text=trial['question'], pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
        question_stim.draw()
        self.window.flip()


        # Collect response (after question display duration)
        key, trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])

        # Check if any key from the specified keys was pressed then check whether it was the correct response (jorn check plz)
        if key in self.const.response_keys:
            trial['response'] = (key == self.const.response_keys[1])
            trial['correct'] = (trial['response'] == trial['answer'])
        else:
            trial['correct'] = False

        # Provide feedback if necessary
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        return trial

class DegradedPassage(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f'{self.name} Task \n\nListen to the audio attentively.'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the task. """

        self.screen.fixation_cross()

        # Load and play audio stimulus for the current trial
        audio_path = self.const.stim_dir / self.name / trial['stim']
        audio_stim = sound.Sound(str(audio_path))
        audio_stim.play()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class IntactPassage(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f'{self.name} Task \n\nListen to the audio attentively.'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the task. """

        self.screen.fixation_cross()

        # Load and play audio stimulus for the current trial
        audio_path = self.const.stim_dir / self.name / trial['stim']
        audio_stim = sound.Sound(str(audio_path))
        audio_stim.play()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class ActionObservation(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

    def display_instructions(self): # overriding the display instruction from the parent class
        self.instruction_text = f"{self.name} Task \n\n Keep your head still while watching the two clips. \n\n Try and remember the knot shown."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Runs a single trial of the ActionObservation task """
        # Assuming that 'stim' column in trial contains the file name of the video clip
        movie_file_name = trial['stim']

        # Construct the movie file path
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / movie_file_name

        # Convert Path object to string for compatibility
        movie_path_str = str(movie_path)

        # Create a MovieStim3 object
        movie_clip = visual.MovieStim3(self.window, movie_path_str, loop=False)

        while movie_clip.status != visual.FINISHED:
            movie_clip.draw()
            self.window.flip()

        self.screen.fixation_cross()

        # Display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class DemandGridEasy(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.grid_size = (3,4)
        self.square_size = 1.5

    def init_task(self):
        """Read the target file and get all the stimuli necessary"""
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')


    def display_instructions(self):
        self.instruction_text = (f"{self.name} Task \n\n"
                                "Watch the sequence of boxes that light \n\n"
                                "up and then choose the correct pattern")
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1],  wrapWidth=400)
        instr_visual.draw()
        self.window.flip()

    def create_grid(self, sequence=None, position='center'):
        """Creates the grid of squares for the DemandGrid task, lighting up specific squares blue if a sequence is given,
        and positions the grid left, right, or center."""

        # Calculate offsets based on the desired position
        if position == 'left':
            offset_x = -10
        elif position == 'right':
            offset_x = 10
        else:  # center
            offset_x = 0

        # Center the grid vertically
        offset_y = 0

        grid = []

        # Create and draw the grid
        for i in range(self.grid_size[0]):
            row = []
            for j in range(self.grid_size[1]):
                # Calculate position with the offsets
                square_x = (j - self.grid_size[0] / 2 + 0.5) * self.square_size + offset_x
                square_y = (self.grid_size[1] / 2 - i - 0.5) * self.square_size + offset_y

                # Determine the fill color based on the sequence
                fill_color = 'blue' if sequence and (i, j) in sequence else 'white'

                rect = visual.Rect(self.window, width=self.square_size, height=self.square_size,
                                pos=(square_x, square_y), lineWidth=3,
                                lineColor='black', fillColor=fill_color)
                rect.draw()
                row.append(rect)
            grid.append(row)

        return grid

    def run_trial(self, trial):
        """Runs a single trial of the DemandGrid task"""

        # Draw the entire grid in its initial state
        self.grid  = self.create_grid()
        self.window.flip()

        # Display the sequence
        original_sequence = literal_eval(trial['grid_sequence'])
        for pos in original_sequence:
            x, y = pos
            self.grid[x][y].fillColor = 'blue'
            for row in self.grid:
                for rect in row:
                    rect.draw()
            self.window.flip()
            core.wait(1)  # Each box lights up for 1 second
            self.grid[x][y].fillColor = 'white'

         # Determine which side the correct sequence will be displayed
        correct_side = trial['correct_side']


        # # Display the original and modified sequence on the left or right side
        modified_sequence = literal_eval(trial['modified_sequence'])

        original_grid = self.create_grid(sequence=original_sequence, position=correct_side)
        modified_grid = self.create_grid(sequence=modified_sequence, position='left' if correct_side == 'right' else 'right')
        self.window.flip()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])

        # Check if any key from the specified keys was pressed then check whether it was the correct response (jorn check plz)
        if key in self.const.response_keys:
            trial['response'] = 'left' if (key == self.const.response_keys[1]) else 'rights'
            trial['correct'] = (trial['response'] == trial['correct_side'])
        else:
            trial['correct'] = False

        # Provide feedback if necessary
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

class DemandGridHard(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.grid_size = (3,4)
        self.square_size = 1.5

    def init_task(self):
        """Read the target file and get all the stimuli necessary"""
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')


    def display_instructions(self):
        self.instruction_text = (f"{self.name} Task \n\n"
                                "Watch the sequence of boxes that light \n\n"
                                "up and then choose the correct pattern")
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1],  wrapWidth=400)
        instr_visual.draw()
        self.window.flip()

    def create_grid(self, sequence=None, position='center'):
        """Creates the grid of squares for the DemandGrid task, lighting up specific squares blue if a sequence is given,
        and positions the grid left, right, or center."""
        # Calculate offsets based on the desired position
        if position == 'left':
            offset_x = -10
        elif position == 'right':
            offset_x = 10
        else:  # center
            offset_x = 0

        # Center the grid vertically
        offset_y = 0

        grid = []

        # Create and draw the grid
        for i in range(self.grid_size[0]):
            row = []
            for j in range(self.grid_size[1]):
                # Calculate position with the offsets
                square_x = (j - self.grid_size[0] / 2 + 0.5) * self.square_size + offset_x
                square_y = (self.grid_size[1] / 2 - i - 0.5) * self.square_size + offset_y

                # Determine the fill color based on the sequence
                fill_color = 'blue' if sequence and (i, j) in sequence else 'white'

                rect = visual.Rect(self.window, width=self.square_size, height=self.square_size,
                                pos=(square_x, square_y), lineWidth=3,
                                lineColor='black', fillColor=fill_color)
                rect.draw()
                row.append(rect)
            grid.append(row)

        return grid


    def run_trial(self, trial):
        """Runs a single trial of the DemandGrid task with two boxes lighting up at a time"""

        # Draw the entire grid in its initial state
        self.grid = self.create_grid()
        self.window.flip()

        # Display the sequence in pairs
        original_sequence = literal_eval(trial['grid_sequence'])
        for i in range(0, len(original_sequence), 2):  # Iterate in steps of 2
            if i + 1 < len(original_sequence):
                pair = [original_sequence[i], original_sequence[i + 1]]
            else:
                pair = [original_sequence[i]]  # In case of an odd number of elements in the sequence

            for pos in pair:
                x, y = pos
                self.grid[x][y].fillColor = 'blue'

            for row in self.grid:
                for rect in row:
                    rect.draw()
            self.window.flip()
            core.wait(1)  # Each pair of boxes lights up for 1 second

            for pos in pair:
                x, y = pos
                self.grid[x][y].fillColor = 'white'

         # Determine which side the correct sequence will be displayed
        correct_side = trial['correct_side']


        # # Display the original and modified sequence on the left or right side
        modified_sequence = literal_eval(trial['modified_sequence'])

        original_grid = self.create_grid(sequence=original_sequence, position=correct_side)
        modified_grid = self.create_grid(sequence=modified_sequence, position='left' if correct_side == 'right' else 'right')
        self.window.flip()

        # collect responses
        key,trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])

        # Check if any key from the specified keys was pressed then check whether it was the correct response (jorn check plz)
        if key in self.const.response_keys:
            trial['response'] = 'left' if (key == self.const.response_keys[1]) else 'rights'
            trial['correct'] = (trial['response'] == trial['correct_side'])
        else:
            trial['correct'] = False

        # Provide feedback if necessary
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        return trial

class SentenceReading(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'None'

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f'{self.name} Task \n\n Read each English word and press a button when the image of a hand pressing a button is displayed'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the sentence reading task. """

        # get sentence and split into words by space
        sentence = trial['stim']
        words = sentence.split()

        #show words seqeuntially each for 450ms
        for word in words:
            word_stim = visual.TextStim(self.window, text=word, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
            word_stim.draw()
            self.window.flip()
            self.ttl_clock.wait_until(self.ttl_clock.get_time() + 0.45)

        # show press button image
        button_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'hand_press_transparent.png'))
        button_stim.draw()
        self.window.flip()
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + 0.4)

        # show blank_transparent image
        blank_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'blank_transparent.png'))
        blank_stim.draw()
        self.window.flip()

        # not displaying crosshair to replicate the localizer

        return trial

class NonwordReading(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f'{self.name} Task \n\n Read each nonword word and press a button when the image of a hand pressing a button is displayed'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the nonword reading task. """

        # get sentence and split into words by space
        sentence = trial['stim']
        words = sentence.split()

        #show words seqeuntially each for 450ms
        for word in words:
            word_stim = visual.TextStim(self.window, text=word, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg')
            word_stim.draw()
            self.window.flip()
            self.ttl_clock.wait_until(self.ttl_clock.get_time() + 0.45)

        # show press button image
        button_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'hand_press_transparent.png'))
        button_stim.draw()
        self.window.flip()
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + 0.4)

        # show blank_transparent image
        blank_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'blank_transparent.png'))
        blank_stim.draw()
        self.window.flip()

        # not displaying crosshair to replicate the localizer

        return trial

class OddBall(Task):
    def __init__(self, info, screen, ttl_clock, const):
        super().__init__(info, screen, ttl_clock, const)
        self.feedback_type = 'acc'

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.target_dir / self.name / self.target_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f'{self.name} Task \n\n Press the button with your index finger when you see a red K'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the oddball task. """

        current_trial =  trial['stim']

        # show stem
        if current_trial == 'red_K':
            word_stim = visual.TextStim(self.window, text='K', pos=(0.0, 0.0), color='red', units='deg', height=1.5)
        elif current_trial == 'black_K':
            word_stim = visual.TextStim(self.window, text='K', pos=(0.0, 0.0), color='black', units='deg', height=1.5)
        elif current_trial == 'red_O':
            word_stim = visual.TextStim(self.window, text='O', pos=(0.0, 0.0), color='red', units='deg', height=1.5)
        elif current_trial == 'black_O':
            word_stim = visual.TextStim(self.window, text='O', pos=(0.0, 0.0), color='black', units='deg', height=1.5)

        word_stim.draw()
        self.window.flip()
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # Show fixation cross
        self.screen.fixation_cross()

        # (need to check with jorn, our feedback method clashes with the task)
    #    # Collect response (after question display duration)
    #     key, trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['iti_dur'])

    #     # Check if any key from the specified keys was pressed then check whether it was the correct response (jorn check plz)
    #     if key in self.const.response_keys:
    #         trial['response'] = (key == self.const.response_keys[1])
    #         trial['correct'] = (trial['response'] == trial['answer'])
    #     else:
    #         trial['correct'] = False

        # # Provide feedback if necessary
        # self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

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