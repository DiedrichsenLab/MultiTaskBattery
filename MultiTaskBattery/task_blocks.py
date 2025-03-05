# Task Class definitions
# March 2021: First version: Ladan Shahshahani  - Maedbh King - Suzanne Witt,
# Revised 2023: Bassel Arafat, Jorn Diedrichsen, Incé Husain
# Revised 2024: Caroline Nettekoven

from pathlib import Path
import pandas as pd
import numpy as np
import random 
from psychopy import visual, sound, core, event
import MultiTaskBattery.utils as ut
from ast import literal_eval
from copy import deepcopy
from moviepy import AudioFileClip
import gc


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

    def __init__(self, info, screen, ttl_clock, const, subj_id):

        # Pointers to Screen and experimental constants
        self.screen             = screen
        self.window             = screen.window # Shortcut to window
        self.const              = const
        self.ttl_clock          = ttl_clock  # This is a reference to the clock of the run
        self.name               = info['task_name']
        self.descriptive_name   = info['descriptive_name']
        self.code               = info['task_code']
        self.task_file          = info['task_file']
        self.feedback_type      = 'none'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')

    def display_instructions(self):
        """
        displays the instruction for the task
        Most tasks have the same instructions. (Tasks that have True/False responses)
        Those tasks that have different instructions will have their own routine
        """
        true_str = f"if True press {self.const.response_keys[1]}"
        false_str = f"if False press {self.const.response_keys[2]}"

        self.instruction_text = f"{self.descriptive_name} Task\n\n {true_str} \n {false_str}"

        # 3.2 display the instruction text
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        # instr.size = 0.8
        instr_visual.draw()
        self.window.flip()
    
    def run(self):
        """Loop over trials and collects data
        Data will be stored in self.trial_data

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
        # Calculate the feedback for the run
        acc = None
        rt = None
        if self.feedback_type[:3] == 'acc':
            acc = self.trial_data['correct'].mean()
        if self.feedback_type[-2:] == 'rt':
            rt = self.trial_data['rt'].mean()
        return acc,rt
    

    def show_progress(self, seconds_left, show_last_seconds=5, height=1, width=10, x_pos=-5, y_pos=8):
        """ Displays a progress bar for the Picture Sequence task
        Args:
            trial (dict): The current trial
            start_time (float): The start time of the trial
            height (float): The height of the progress bar
            width (float): The width of the progress bar
            y_pos (float): The y position of the progress bar
        """
        # If we are in the last five seconds of the trial, display the remaining time
        if seconds_left < show_last_seconds:
            progress = visual.Progress(
                win=self.window, 
                progress=1-(seconds_left/show_last_seconds),
                size=(width, height),
                pos=(x_pos, y_pos),
                backColor='blue',
                barColor='black',
                borderColor='black',
                lineWidth=5,
            )
            progress.draw()

    def wait_response(self, start_time, max_wait_time, show_last_seconds=None, current_stimuli=None):
        """
        waits for a response to be made and then returns the response
        Args:
            start_time (float): the time the RT-period started
            max_wait_time (float): How long to wait maximally
        Returns:
            key (str): the key that was pressed (1-4) (0 if no key was pressed)
            rt (float): the reaction time (nan if no key was pressed)
        """
        response_made = False
        key = 0
        rt = np.nan

        while (self.ttl_clock.get_time() - start_time <= max_wait_time) and not response_made:
            self.ttl_clock.update()
            if show_last_seconds is not None:
                current_stimuli.draw()
                seconds_left = max_wait_time - (self.ttl_clock.get_time() - start_time)
                self.show_progress(seconds_left,
                                show_last_seconds=show_last_seconds,
                                y_pos=5)
                self.window.flip()
            keys=event.getKeys(keyList= self.const.response_keys, timeStamped=self.ttl_clock.clock)
            if len(keys)>0:
                response_made = True
                key_char = keys[0][0]
                key = self.const.response_keys.index(key_char) + 1
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
                self.screen.check_mark('green')
            else:
                self.screen.error_cross('red')
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

    def screen_quit(self):
        """ Checks for quit or escape key presses and quits the experiment if necessary """
        keys = event.getKeys()
        for key in keys:
            if 'q' and 'esc' in key:
                self.window.close()
                core.quit()

    def get_audio_from_movie(self, movie_path, sample_rate=48000):
        """Seperates the audio from the movie file and returns the audio object (for better memory handling when playing movies with sound)"""

        # Gets the movie audio
        audio_clip = AudioFileClip(movie_path)
        audio_array = audio_clip.to_soundarray(fps=sample_rate)
        audio_clip.close()
        audio = sound.Sound(audio_array,sampleRate=sample_rate, stereo=True)

        return audio

class NBack(Task):

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')
        self.stim=[]
        for stim in self.trial_info['stim']:
            stim_path = self.const.stim_dir / self.name / stim
            self.stim.append(visual.ImageStim(self.window, str(stim_path)))
        self.corr_key = [self.trial_info['key_nomatch'].iloc[0],self.trial_info['key_match'].iloc[0]]

    def display_instructions(self):
        """
        displays the instruction for the task
        """
        str1 = f"Compare image to the one shown 2 previously"
        str2 = f"if match, press {self.corr_key[1]}"
        str3 = f"if no match, press {self.corr_key[0]}"
        self.instruction_text = f"{self.descriptive_name} Task\n\n {str1} \n {str2} \n {str3}"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

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

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])
        trial['correct'] = (trial['response'] == self.corr_key[trial['trial_type']])

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])
        return trial

class Rest(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
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
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')
        self.trial_info['noun'] = self.trial_info['stim'].str.strip()

    def display_instructions(self): # overriding the display instruction from the parent class

        self.instruction_text = f"{self.descriptive_name} Task \n\n Silently read the words presented.  \n\n When GENERATE is shown, silently think of verbs that go with the words."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def show_stim(self, noun):
        """ Display a word for a fixed time. """
        stim = visual.TextStim(self.window, text=noun, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=2)
        stim.draw()
        self.window.flip()

    def display_generate_instruction(self):
        """ Display the 'GENERATE' instruction. """
        generate_instr = visual.TextStim(self.window, text='GENERATE', pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=2)
        generate_instr.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Run a single trial of the VerbGeneration task. """

        # Display word
        self.show_stim(trial['noun'])

        # display GENERATE instruction at the halfway point (jorn check this plz, this replaces word 7 with generate)
        if trial.name == len(self.trial_info) // 2:
            self.display_generate_instruction()

        # wait for trial duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)
        return trial

class TongueMovement(Task):
    """
    Tongue movement following Buckner et al., 2022! No particular feedback.
    """
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self):
        self.instruction_text = f"{self.descriptive_name} Task \n\n Move your tongue left to right touching your upper premolar teeth"
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
            circle_visual = visual.Circle(self.window, radius=3, edges= 100, lineWidth = 20, fillColor=None, lineColor='black')
            circle_visual.draw()

        self.window.flip()

        # wait for trial duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class AuditoryNarrative(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self):
        self.instruction_text = f'{self.descriptive_name} Task\n\nListen to the narrative attentively.'
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

        trial_dur = audio_stim.getDuration()

        # wait for trial duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class SpatialNavigation(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)


    def display_instructions(self):
        start_location = self.trial_info.iloc[0]['location_1']
        end_location = self.trial_info.iloc[0]['location_2']

        self.instruction_text = (f"{self.descriptive_name} Task \n\n"
                                    f"Imagine walking around your childhood home\n"
                                    f"Start in the {start_location} – end in the {end_location}\n"
                                    f"Focus on the fixation cross")
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1],  wrapWidth=20)
        instr_visual.draw()
        self.window.flip()

    def run_trial(self,trial):
        self.screen.fixation_cross()

        # wait for trial duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class TheoryOfMind(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['key_false'].iloc[0],self.trial_info['key_true'].iloc[0]]

        
    def display_instructions(self):
        """
        displays the instruction for the task
        """
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()
        str1 = f"You will read a story and decide if the answer to the question is True or False."
        str2 = f"if true, press {self.corr_key[1]}"
        str3 = f"if false, press {self.corr_key[0]}"
        self.instruction_text = f"\n\n {str1} \n\n {str2} \n {str3}"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Runs a single trial of the Theory of Mind task """

        event.clearEvents()

        # Display story
        story_stim = visual.TextStim(self.window, text=trial['story'], alignHoriz='center', wrapWidth=20, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height= 1.25)
        story_stim.draw()
        self.window.flip()

       # wait until story duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['story_dur'])

        # Flush any keys in buffer
        event.clearEvents()

        # Display question
        question_stim = visual.TextStim(self.window, text=trial['question'], pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=25)
        question_stim.draw()
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])
        trial['correct'] = (trial['response'] == self.corr_key[trial['trial_type']])

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        return trial

class DegradedPassage(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self):
        self.instruction_text = f'{self.descriptive_name} Task \n\nListen to the audio attentively.'
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

        # wait for trial duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class IntactPassage(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self):
        self.instruction_text = f'{self.descriptive_name} Task \n\nListen to the audio attentively.'
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

        # wait for trial duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial

class ActionObservation(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self): # overriding the display instruction from the parent class
        self.instruction_text = f"{self.descriptive_name} Task \n\n Keep your head still while watching the two clips. \n\n Try and remember the knot shown."
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
        movie_clip = visual.MovieStim(self.window, movie_path_str, loop=False)

        while movie_clip.isFinished == False:
            movie_clip.play()
            movie_clip.draw()
            self.window.flip()
            self.ttl_clock.update()

        self.screen.fixation_cross()

        # Display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        # Flush memory
        movie_clip.unload()
        gc.collect() # Collect garbarge

        return trial

class DemandGrid(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.square_size = 1.5
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')
        self.corr_key = [self.trial_info['key_left'].iloc[0],self.trial_info['key_right'].iloc[0]]

    def display_instructions(self):
        """
        displays the instruction for the task
        """
        str1 = f"You will watch the sequence of boxes that light up and then choose the correct pattern"
        str2 = f"if left, press {self.corr_key[0]}"
        str3 = f"if right, press {self.corr_key[1]}"
        self.instruction_text = f"{self.descriptive_name} Task\n\n {str1} \n {str2} \n {str3}"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def create_grid(self, sequence=None, position='center',grid_size=(3,4)):
        """Creates the grid of squares for the DemandGrid task, lighting up specific squares blue if a sequence is given,
        and positions the grid left, right, or center."""
        # Calculate offsets based on the desired position
        if position == 'left':
            offset_x = -5
        elif position == 'right':
            offset_x = 5
        else:  # center
            offset_x = 0

        # Center the grid vertically
        offset_y = 0

        grid = []
        # Create and draw the grid
        for i in range(grid_size[0]):
            row = []
            for j in range(grid_size[1]):
                # Calculate position with the offsets
                square_x = (j - grid_size[0] / 2 + 0.5) * self.square_size + offset_x
                square_y = (grid_size[1] / 2 - i - 0.5) * self.square_size + offset_y

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
        if 'grid_size' in trial:
            grid_size = literal_eval(trial['grid_size'])
        else:
            grid_size = (3,4)

        # Make the code adaptable to old DemandGrid implementation
        if 'num_steps' in trial:
            num_steps = trial['num_steps']
        else:
            num_steps = 3

        step_dur = trial['sequence_dur']/num_steps
        self.grid = self.create_grid(grid_size=grid_size)
        self.window.flip()

        # Display the sequence in steps
        if 'original_sequence' in trial:
            original_sequence = literal_eval(trial['original_sequence'])
        else:
            original_sequence = literal_eval(trial['grid_sequence'])

        if 'num_steps' in trial: # new implementation
            for i in range(num_steps):
                step_sequence_name = f'original_step_{i+1}'
                step_sequence = literal_eval(trial[step_sequence_name])

                for tuple in step_sequence:
                    x, y = tuple
                    self.grid[x][y].fillColor = 'blue'

                for row in self.grid:
                    for rect in row:
                        rect.draw()
                self.window.flip()
                self.ttl_clock.wait_until(self.ttl_clock.get_time() + step_dur)

                for tuple in step_sequence:
                    x, y = tuple
                    self.grid[x][y].fillColor = 'white'

        else: # old implementation
            for i in range(0, len(original_sequence), 2):  # Iterate in steps of 2
                if i + 1 < len(original_sequence):
                    pair = [original_sequence[i], original_sequence[i + 1]]
                else:
                    pair = [original_sequence[i]]  # Handle odd-length sequences

                # Highlight positions in the current pair
                for x, y in pair:
                    self.grid[x][y].fillColor = 'blue'

                # Draw and update the window
                for row in self.grid:
                    for rect in row:
                        rect.draw()
                self.window.flip()
                self.ttl_clock.wait_until(self.ttl_clock.get_time() + step_dur)

                # Reset colors after the pair
                for x, y in pair:
                    self.grid[x][y].fillColor = 'white'

        # Flush any keys in buffer
        event.clearEvents()

         # Determine which side the correct sequence will be displayed
        correct_side = trial['correct_side']


        # # Display the original and modified sequence on the left or right side
        modified_sequence = literal_eval(trial['modified_sequence'])

        original_grid = self.create_grid(sequence=original_sequence, position=correct_side, grid_size=grid_size)
        modified_grid = self.create_grid(sequence=modified_sequence, position='left' if correct_side == 'right' else 'right', grid_size=grid_size)
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])
        trial['correct'] = (trial['response'] == self.corr_key[trial['trial_type']])

        # Provide feedback if necessary
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        return trial

class SentenceReading(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self):
        self.instruction_text = f'{self.descriptive_name} Task \n\n Read each English word and press a button when the image of a hand pressing a button is displayed'
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
            word_stim = visual.TextStim(self.window, text=word, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=2)
            word_stim.draw()
            self.window.flip()
            self.ttl_clock.wait_until(self.ttl_clock.get_time() + 0.45)
        
        event.clearEvents()

        # show press button image
        button_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'hand_press_transparent.png'))
        button_stim.draw()
        self.window.flip()
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), 0.4)

        # show blank_transparent image
        blank_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'blank_transparent.png'))
        blank_stim.draw()
        self.window.flip()

        # flush any keys in buffer
        event.clearEvents()

        # not displaying crosshair to replicate the localizer

        return trial

class NonwordReading(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self):
        self.instruction_text = f'{self.descriptive_name} Task \n\n Read each nonword word and press a button when the image of a hand pressing a button is displayed'
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
            word_stim = visual.TextStim(self.window, text=word, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=2)
            word_stim.draw()
            self.window.flip()
            self.ttl_clock.wait_until(self.ttl_clock.get_time() + 0.45)

        event.clearEvents()

        # show press button image
        button_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'hand_press_transparent.png'))
        button_stim.draw()
        self.window.flip()
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), 0.4)

        # show blank_transparent image
        blank_stim = visual.ImageStim(self.window, image=str(self.const.stim_dir / self.name / 'blank_transparent.png'))
        blank_stim.draw()
        self.window.flip()

        # clear any keys in buffer
        event.clearEvents()

        # not displaying crosshair to replicate the localizer

        return trial

class OddBall(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')  
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0]]  

    def display_instructions(self):
        """
        displays the instruction for the task
        """
        str1 = f"Press {self.corr_key[0]} when you see a red K"
        self.instruction_text = f"{self.descriptive_name} Task\n\n {str1} \n"
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

        self.display_trial_feedback(trial['display_trial_feedback'], None)

        # Flush any keys in buffer
        event.clearEvents()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['iti_dur'])

        # method 2 for accuracy for oddball
        if trial['trial_type'] == 1:
            trial['correct'] = True if trial['response']!= 0 else False
        elif trial['trial_type'] == 0:
            trial['correct'] = np.nan


        return trial
    
class FingerSequence(Task):
    """
    Finger sequence task
    """
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0],self.trial_info['key_three'].iloc[0],self.trial_info['key_four'].iloc[0]]


    def display_instructions(self):
        self.instruction_text = f"{self.descriptive_name} Task \n\n Using your four fingers, press the keys in the order shown on the screen\n Use all four fingers for this task"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()


    def run_trial(self, trial):
        """ Run a single trial of the finger sequence task. """
        #clear buffer
        event.clearEvents()

         # Display the sequence
        sequence = trial['stim'].split()

        # Calculate the start position for the sequence and determine the spacing between numbers
        num_items = len(sequence)
        spacing = 2.0  
        start_x = -(num_items - 1) * spacing / 2

        # Show the numbers in the sequence next to each other ( using the spacing and start_x calculated above)
        for i, number in enumerate(sequence):
            pos = (start_x + i * spacing, 0.0)  # Horizontal position is adjusted based on index
            stim = visual.TextStim(self.window, text=number, pos=pos, color='black', units='deg', height=1.5)
            stim.draw()

        self.window.flip()

        
        sequence_start_time = self.ttl_clock.get_time() # Needed for knowing when to stop looking for key presses
        digit_start_time = sequence_start_time # Updated with each key press for calculating RT

        rt_list = np.full(num_items,np.nan)
        correct_list = np.zeros((num_items,)) # List of booleans indicating whether each press was correct needed for overall trial accuracy
        num_presses =0
        # Initialize the color for each digit in the sequence as black
        digit_colors = ['black'] * num_items
        while self.ttl_clock.get_time() - sequence_start_time < trial['trial_dur'] and num_presses < num_items:
            self.ttl_clock.update()

            keys = event.getKeys(keyList=self.const.response_keys, timeStamped=self.ttl_clock.clock)
            if keys:
                key_char, key_press_time = keys[0]
                key = self.const.response_keys.index(key_char) + 1
                rt = key_press_time - digit_start_time
                rt_list[num_presses]=rt
                digit_start_time = key_press_time

                # Check if key pressed is correct
                correct_list[num_presses] = key == int(sequence[num_presses])
 
                # Update color based on correctness
                digit_colors[num_presses] = 'green' if correct_list[num_presses] else 'red'

                num_presses += 1
            # Draw all digits with their adjusted colors
            for i, (number, color) in enumerate(zip(sequence, digit_colors)):
                pos = (start_x + i * spacing, 0.0)
                stim = visual.TextStim(self.window, text=number, pos=pos, color=color, units='deg', height=1.5)
                stim.draw()

            self.window.flip()

        else:
            # If the sequence is completed, wait until the end of the trial
            self.ttl_clock.wait_until(sequence_start_time + trial['trial_dur'])

        # if any press is wrong trial['correct'] needs to be false, this is for post trial feedback
        trial['correct'] = correct_list.sum()/num_items

        if np.all(np.isnan(rt_list)):
            # calculate mean rt across presses
            trial['rt'] = np.nan

        else:
            trial['rt'] = np.nanmean(rt_list)
 
        # display trial feedback (for whole trial)
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct']== 1)

        return trial
    
class Sencoding(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)

    def display_instructions(self):
        self.instruction_text = f'{self.descriptive_name} Task \n\nListen to the following sentences attentively.'
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

        # wait for trial duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])

        # display trial feedback
        self.display_trial_feedback(give_feedback= trial['display_trial_feedback'], correct_response = None)

        return trial
    
class SencodingProbe(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['first_key'].iloc[0],self.trial_info['second_key'].iloc[0]]

        
    def display_instructions(self):
        """
        displays the instruction for the task
        """
        str1 = f"You will read sentences and decide which completion is closer to a sentence you heard in the last run"
        self.instruction_text = f"{self.descriptive_name} Task\n\n {str1}"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Runs a single trial of the Theory of Mind task """

        event.clearEvents()

        # Display story
        str1 = trial['stem']
        str2 = trial['option_1']
        str3 = trial['option_2']

        story_stim = visual.TextStim(self.window, text=f'Stem:{str1} \n\n A: {str2} \n B:{str3}', alignHoriz='center', wrapWidth=25, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height= 1.25)
        story_stim.draw()
        self.window.flip()

       # wait until story duration
        # self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'] + trial['iti_dur'])


        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])

        if trial['answer'] == trial['option_1']:
            trial['trial_type'] = 0
        elif trial['answer'] == trial['option_2']:
            trial['trial_type'] = 1
        else:
            print('answer doesnt match either option check if there is lagging space')
            
        trial['correct'] = (trial['response'] == self.corr_key[trial['trial_type']])

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        # Flush any keys in buffer
        event.clearEvents()

        return trial
                           
class FlexionExtension(Task):
    """
    Flexion extension of toes! No particular feedback.
    """
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'None'

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')

    def display_instructions(self):
        self.instruction_text = f"{self.descriptive_name} Task \n\n Flex and extend your right and left toes"
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
    
    #semantic prediction runs (slight bug for response feedback: last word is not synced with last word in task_file..)

class SemanticPrediction(Task):

    """
    Read a sentence and decide if the last word of the sentence makes sense. Click "3" if the last word makes sense; click "4" if not. Be as accurate and fast as possible.
    """

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['key_false'].iloc[0],self.trial_info['key_true'].iloc[0]] 

    def display_instructions(self):
        """
        displays the instruction for the task
        """
        str1 = f"You will read a sentence and decide if the last word makes sense."
        str2 = f"If it makes sense, press {self.corr_key[1]}"
        str3 = f"if it doesn't make sense, press {self.corr_key[0]}"
        self.instruction_text = f"{self.descriptive_name} Task\n\n {str1} \n {str2} \n {str3}"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()
    
    def run_trial(self, trial):
        """ Runs a single trial of the semantic prediction task """
        
        height_word = 2 

        event.clearEvents()

        # get sentence and split into words by space
        sentence = trial['sentence']
        words = sentence.split('|')

        #show words seqeuntially each for 800ms
        for word in words:
            word_stim = visual.TextStim(self.window, text=word, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=height_word)
            word_stim.draw()
            self.window.flip()
            self.ttl_clock.wait_until(self.ttl_clock.get_time() + 0.8)

        event.clearEvents()

        # Display last word
        last_word_stim = visual.TextStim(self.window, text=trial['last_word'], pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height= height_word, wrapWidth=30)
        last_word_stim.draw()
        self.window.flip()

        event.clearEvents()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['sentence_dur'])
        trial['correct'] = (trial['response'] == self.corr_key[trial['trial_type']])

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        return trial
    
class VisualSearch(Task):

    """
    Look at a screen filled with shapes and identify whether an "L" is present. Click "3" if the "L" is present; click "4" if not. Be as accurate and fast as possible.
    """

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')

        #counter initialized in order to read whether trial_type is true or not for each trial; used in generate_trial_stimuli 
        self.trial_counter = 0 
        
        #width and height determined by trial and error: x extremity of window is 14; y extremity is 10 
        screen_width = 28
        screen_height = 20

        #define number of rows + col to split screen into 24 apertures
        num_rows = 4
        num_cols = 6

        #Define aperture sizes + positions 
        aperture_width = screen_width / num_cols
        aperture_height = screen_height / num_rows

        positions_x = np.linspace(-screen_width / 2 + aperture_width / 2, screen_width / 2 - aperture_width / 2, num_cols)
        positions_y = np.linspace(-screen_height / 2 + aperture_height / 2, screen_height / 2 - aperture_height / 2, num_rows)

        aperture_positions = []

        for y in positions_y:
            for x in positions_x:
                aperture_positions.append((x, y))
        
        self.apertures = []
        for pos in aperture_positions:
            apertures = visual.Aperture(self.window, size=40, shape = 'rectangle', pos=pos, units='norm')
            self.apertures.append(apertures)

    def generate_trial_stimuli(self, num_stimuli):

        stim_images = ['90.png','180.png','270.png','360.png']

        self.stim = []
        
        is_target = True #determines whether target will appear on screen 

        if self.trial_info['trial_type'][self.trial_counter] == 0: 
            stim_images.remove('90.png') #sets a display with no target 
            is_target = False 

        self.trial_counter+=1

        randomly_select_apertures = random.sample(self.apertures, num_stimuli)
        randomly_select_apertures = random.sample(self.apertures, num_stimuli)

        for aperture in randomly_select_apertures:
            if is_target:
                stim_current = stim_images[0] #sets a display with at least one target 
                is_target = False 
            else: 
                stim_random_idx = random.randint(0, len(stim_images)-1)  #chooses random stimuli from stim_images list 
                stim_current = stim_images[stim_random_idx]

            stim_path = self.const.stim_dir/ self.name / stim_current
            stimulus = visual.ImageStim(self.window, str(stim_path), size=(0.8,0.8))
            stimulus.setPos([aperture.pos[0], aperture.pos[1]]) #puts stimuli within apertures 
            self.stim.append(stimulus) #creates list of all randomly selected stimuli 

    def display_instructions(self):
        """
        displays the instruction for the task
        """

        self.corr_key = [self.trial_info['key_false'].iloc[0],self.trial_info['key_true'].iloc[0]]

        str1 = f"You will survey a series of shapes and identify whether the letter ‘L’ is present."
        str2 = f"If 'L' is present, press {self.corr_key[1]}"
        str3 = f"if 'L' is not present, press {self.corr_key[0]}"
        self.instruction_text = f"{self.descriptive_name} Task\n\n {str1} \n {str2} \n {str3}"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run_trial(self,trial):
        """Runs a single trial of visual search task
        """
        
        # Flush any keys in buffer
        event.clearEvents()
        
        num_stimuli = self.trial_info.loc[trial['trial_num'], 'num_stimuli']  #indicates if trial is easy (4 stimuli) or hard (8 stimuli) 

        self.generate_trial_stimuli(num_stimuli)

        self.screen.fixation_cross(flip=False)  #show fixation cross in center of screen

        # Display stimuli
        for stimulus in self.stim:
            stimulus.draw()

        self.window.flip()

        # collect responses 
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])
        trial['correct'] = (trial['response'] == self.corr_key[trial['trial_type']])
        
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        return trial


class RMET(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0],self.trial_info['key_three'].iloc[0],self.trial_info['key_four'].iloc[0]]

    def display_instructions(self):
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()
        self.instruction_text = ""
        if 'age' in self.task_file:
            self.instruction_text += "\n\n Choose what AGE the person is."
        elif 'emotion' in self.task_file:
            self.instruction_text += "\n\n Choose what FEELING the person has."
        else:
            self.instruction_text += "\n\n Choose which AGE or FEELING best describes the person." # General instruction for both age and emotion
        self.instruction_text += f"\n\n\n{self.trial_info['key_one'].iloc[0]}. index \t{self.trial_info['key_two'].iloc[0]}. middle\t{self.trial_info['key_three'].iloc[0]}. ring\t{self.trial_info['key_four'].iloc[0]}. pinky"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 0))
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Runs a single trial of the Reading the Mind in the Eye (RMET) task """
        
        # Flush any keys in buffer
        event.clearEvents()
        
        # --- Eyes ---
        # Get the file name
        picture_file_name = trial['stim']
        # Construct the picture file path
        picture_path = Path(self.const.stim_dir) / self.name / 'pictures' / picture_file_name
        # Convert Pathself object to string for compatibility
        picture_path_str = str(picture_path)
        # Create an ImageStim object
        picture = visual.ImageStim(self.window, str(picture_path_str))
        # Make the picture smaller
        picture_scale = self.const.rmet_picture_scale if hasattr(self.const, 'rmet_picture_scale') else 0.7
        picture.size = picture.size * picture_scale

        

        # --- Answers ---
        # Get the answer options
        answer_options = trial['options']
        # Separate them into four strings
        answer_options = answer_options.split(',')
        # Create TextStim objects for each answer option
        answer_stims = []
        for i, option in enumerate(answer_options):
            # 0 and 1 should be on the left and right of the top line (y position 7 and x positions -7 and 7)
            # 2 and 3 should be on the left and right of the bottom line (y position -7 and x positions -7 and 7)
            x = -8 if i % 2 == 0 else 6
            y = 5 if i < 2 else -5
            
            if len (option) < 3:
                tabs = 2
            elif len(option) < 9:
                tabs = 3
            else:
                tabs = 4
            tab_string = ''.join(["\t"] * tabs)
            answer_stim = visual.TextStim(self.window, text=f'{i+1}.{tab_string}',
                              pos=(x, y-0.04), color='blue', height=1, alignHoriz='center')

            answer_stims.append(answer_stim)
            tab_string = ''.join(["\t"] * (tabs-1))
            answer_stim = visual.TextStim(self.window, text=f'{tab_string}{option}',
                                            pos=(x, y), color=[-1, -1, -1], height=1.4, alignHoriz='center')
            answer_stims.append(answer_stim)

        # Display stimuli
        picture.draw()
        for answer_stim in answer_stims:
            answer_stim.draw()
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])
        trial['correct'] = (trial['response'] == answer_options.index(str(trial['answer']))+1)
        
        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        # Flush any keys in buffer
        event.clearEvents()

        return trial

class PictureSequence(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0],self.trial_info['key_three'].iloc[0],self.trial_info['key_four'].iloc[0]]

    def display_instructions(self):
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()
        self.instruction_text = ""
        self.instruction_text += "\n\n Find the correct chronological order of the pictures."
        self.instruction_text += f"\n\n\n{self.trial_info['key_one'].iloc[0]}. index \t{self.trial_info['key_two'].iloc[0]}. middle\t{self.trial_info['key_three'].iloc[0]}. ring\t{self.trial_info['key_four'].iloc[0]}. pinky"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 0))
        instr_visual.draw()
        self.window.flip()
        
    def show_presses(self, pressed_keys, positions, last_key_press_time, width=1.4, height=7, line_width=10):
        """ Displays the presses on the screen
        Args:
            pressed_keys (list): A list of the keys that have been pressed
            positions (list): A list of the positions of the images
            width (float): The width of the images
            height (float): The height of the images
            line_width (float): The width of the border
            last_key_press_time (float): The time of the last key press
        """
        #Add a black border around the selected images
        for p, pressed_key in enumerate(pressed_keys):
            color = 'blue' if p == len(pressed_keys) - 1 and not self.ttl_clock.get_time() - last_key_press_time > 1  else 'black' #Add a green border around the last selected image if the last key press was less than 2 seconds ago
            visual.Rect(self.window, size=(width, height), pos=positions[pressed_key-1], lineColor=color, lineWidth=line_width).draw()
        
    def run_trial(self, trial):
        """ Runs a single trial of the Reading the Mind in the Eye (RMET) task """
        
        # Flush any keys in buffer
        event.clearEvents()
        
        # Get the file name
        picture_file_name = trial['stim']
        # Construct the picture file path
        picture_paths = [str(Path(self.const.stim_dir) / self.name / 'pictures' / f"{picture_file_name} card{n}") for n in range(1,5)]  
        # Sort them in the order they should be displayed
        sequence = list(map(int, trial['sequence'].split(' ')))
        picture_paths = [picture_paths[i-1] for i in sequence]

        # Define positions for a 2x2 grid layout
        height = 7    
        width = 1.4*height
        x_pos = 5
        y_pos = 3.6
        positions = [
            (-x_pos, y_pos),  # Top-left
            (x_pos, y_pos),   # Top-right
            (-x_pos, -y_pos), # Bottom-left
            (x_pos, -y_pos)   # Bottom-right
        ]
        # Create ImageStim objects for each picture

        pictures = [visual.ImageStim(self.window, image=path, pos=pos, size=(1.4*height, height)) for path, pos in zip(picture_paths, positions)]


        # --- Answers ---
        # Create TextStim objects for each answer option
        answer_options = ['1', '2', '3', '4']
        answer_stims = []
        for i, option in enumerate(answer_options):
            x = -x_pos-width*0.4 if i % 2 == 0 else x_pos+width*0.4
            y = y_pos+height*0.4 if i < 2 else -y_pos-height*0.4
            answer_stim = visual.TextStim(self.window, text=f'{option}', pos=(x, y), color=[-1, -1, -1], height=1.3, alignHoriz='center')
            answer_stims.append(answer_stim)

        # Calculate the start position for the sequence and determine the spacing between numbers
        num_items = len(sequence)
        
        # collect responses 0: no response 1-4: key pressed
        sequence_start_time = self.ttl_clock.get_time() # Needed for knowing when to stop looking for key presses
        digit_start_time = sequence_start_time # Updated with each key press for calculating RT

        rt_list = np.full(num_items,np.nan)
        correct_list = np.zeros((num_items,)) # List of booleans indicating whether each press was correct needed for overall trial accuracy
        num_presses =0
        pressed_keys = []
        line_width = 15
        
        while self.ttl_clock.get_time() - sequence_start_time < trial['trial_dur']:
            self.ttl_clock.update()

            for picture in pictures:
                picture.draw()
            for answer_stim in answer_stims:
                answer_stim.draw()
            
            seconds_left = trial['trial_dur'] - (self.ttl_clock.get_time() - sequence_start_time)
            self.show_progress(seconds_left,
                               show_last_seconds=5,
                               height=1,
                               width=width,
                               x_pos=0-width*0.5,
                               y_pos=y_pos+height*0.5+1)
            self.show_presses(pressed_keys, positions, digit_start_time, width, height, line_width)
            self.window.flip()

            if num_presses < num_items:
                keys = event.getKeys(keyList=self.const.response_keys, timeStamped=self.ttl_clock.clock)
                if keys:
                    key_char, key_press_time = keys[0]
                    key = self.const.response_keys.index(key_char) + 1
                    rt = key_press_time - digit_start_time
                    rt_list[num_presses]=rt
                    digit_start_time = key_press_time

                    # Check if key pressed is correct
                    correct_list[num_presses] = key == int(sequence[num_presses])
                    num_presses += 1
                    pressed_keys.append(key)
            
        # if any press is wrong trial['correct'] needs to be false, this is for post trial feedback
        trial['correct'] = correct_list.sum()/num_items
        trial['response'] = pressed_keys

        if np.all(np.isnan(rt_list)):
            # calculate mean rt across presses
            trial['rt'] = np.nan

        else:
            trial['rt'] = np.nanmean(rt_list)
 
        # display trial feedback (for whole trial)
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct']==1)

        return trial


class StorySequence(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0],self.trial_info['key_three'].iloc[0],self.trial_info['key_four'].iloc[0]]

    def display_instructions(self):
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()
        self.instruction_text = ""
        self.instruction_text += "\n\n Find the correct chronological order of the sentences."
        self.instruction_text += f"\n\n\n{self.trial_info['key_one'].iloc[0]}. index \t{self.trial_info['key_two'].iloc[0]}. middle\t{self.trial_info['key_three'].iloc[0]}. ring\t{self.trial_info['key_four'].iloc[0]}. pinky"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 0))
        instr_visual.draw()
        self.window.flip()
        
    def show_presses(self, sentences, positions, pressed_keys, last_key_press_time, wrapWidth, text_height=1):
        """ Displays the presses on the screen
        Args:
            sentences (list): A list of the sentences to be displayed
            positions (list): A list of the positions of the sentences
            pressed_keys (list): A list of the keys that have been pressed
            last_key_press_time (float): The time of the last key press
            wrapWidth (float): The width of the text
        """
       
        
        for p, pressed_key in enumerate(pressed_keys):
            color = 'blue' if p == len(pressed_keys) - 1 and not self.ttl_clock.get_time() - last_key_press_time > 1  else 'darkgrey' # Present the stimuli in blue if the last key press was less than 2 seconds ago
            visual.TextStim(self.window, text=sentences[pressed_key-1], pos=positions[pressed_key-1], color=color, height=text_height, wrapWidth=wrapWidth).draw()

            
        
    def run_trial(self, trial):
        """ Runs a single trial of the Reading the Mind in the Eye (RMET) task """
        
        # Flush any keys in buffer
        event.clearEvents()
        
        wrapWidth = 20
        
        # Sort them in the order they should be displayed
        sequence = list(map(int, trial['sequence'].split(' ')))
        sentences = [trial[f"stim{i}"] for i in range(1,5)]
        sentences = [sentences[i-1] for i in sequence] # Order the sentences according to the sequence
        # Format the sentences for display
        sentences = [f'{s+1}.\t{sentence}\n\n' for s, sentence in enumerate(sentences)]
        sentences_stim = visual.TextStim(self.window, text=''.join(sentences), pos=(0, 0), color=[-1, -1, -1], height=1, wrapWidth=wrapWidth) 

        # Calculate the start position for the sequence and determine the spacing between numbers
        num_items = len(sequence)
        
        # collect responses 0: no response 1-4: key pressed
        sequence_start_time = self.ttl_clock.get_time() # Needed for knowing when to stop looking for key presses
        digit_start_time = sequence_start_time # Updated with each key press for calculating RT

        rt_list = np.full(num_items,np.nan)
        correct_list = np.zeros((num_items,)) # List of booleans indicating whether each press was correct needed for overall trial accuracy
        num_presses =0
        pressed_keys = []
        line_width = 10
        bar_width = 10
        height = 7
        text_height = 1
        y_pos = 5

        # Arrange sentences non-overlapping from top to bottom
        positions = [(0, 5), (0, 1), (0, -2), (0, -6)]
        # Present the stimuli in black
        
        while self.ttl_clock.get_time() - sequence_start_time < trial['trial_dur']:
            self.ttl_clock.update()
                        
            seconds_left = trial['trial_dur'] - (self.ttl_clock.get_time() - sequence_start_time)
            self.show_progress(seconds_left,
                                show_last_seconds=5,
                                height=1,
                                width=bar_width,
                                x_pos=0-bar_width*0.5,
                                y_pos=y_pos+height*0.5+1)
            # Display the sentences
            [visual.TextStim(self.window, text=sentence, pos=positions[s], color='black', height=text_height, wrapWidth=wrapWidth).draw() for s, sentence in enumerate(sentences)]
            self.show_presses(sentences, positions, pressed_keys, digit_start_time, wrapWidth, text_height)
            self.window.flip()

            if num_presses < num_items:
                keys = event.getKeys(keyList=self.const.response_keys, timeStamped=self.ttl_clock.clock)
                if keys:
                    key_char, key_press_time = keys[0]
                    key = self.const.response_keys.index(key_char) + 1
                    rt = key_press_time - digit_start_time
                    rt_list[num_presses]=rt
                    digit_start_time = key_press_time

                    # Check if key pressed is correct
                    correct_list[num_presses] = key == int(sequence[num_presses])
                    num_presses += 1
                    pressed_keys.append(key)
            
        # if any press is wrong trial['correct'] needs to be false, this is for post trial feedback
        trial['correct'] = correct_list.sum()/num_items

        if np.all(np.isnan(rt_list)):
            # calculate mean rt across presses
            trial['rt'] = np.nan

        else:
            trial['rt'] = np.nanmean(rt_list)
 
        # display trial feedback (for whole trial)
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct']==1)

        return trial
    
class ActionPrediction(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0]]
        
    def display_instructions(self):
        """
        displays the instruction for the task
        """
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()
        self.instruction_text = ""
        if 'soccer' in self.task_file:
            self.instruction_text += "\n\n Decide if the ball is going to the left or right."
            self.instruction_text += f"\n\n\nLEFT: index finger \tRIGHT: middle finger\n"
        elif 'greeting' in self.task_file:
            self.instruction_text += "\n\n Decide if the people will hug or shake hands."
            self.instruction_text += f"\n\n\nHUG: index finger \tSHAKE HANDS: middle finger\n"
        else:
            self.instruction_text += "\n\n Choose where the ball will land or how the people will greet each other." # General instruction for both age and emotion
            self.instruction_text += f"\n\n\nLEFT/HUG: index finger \tRIGHT/SHAKE HANDS: middle finger\n"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=25, pos=(0, 0))
        instr_visual.draw()
        self.window.flip()


    def run_trial(self, trial):
        """ Runs a single trial of the Action Prediction task """

        event.clearEvents()

        window_width, _ = self.window.size
        movie_scale = self.const.movie_scale if hasattr(self.const, 'action_prediction_scale') else 0.4
        stim_width = int(window_width * movie_scale) # Make the video fraction of the window width
        stim_height = int(stim_width  * 476 / 846)  # Original size of the video is 640x360
        
        
        # Display video        
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / f"{trial['stim']}.mp4"
        movie_path_str = str(movie_path)
        movie_clip = visual.MovieStim(self.window, movie_path_str, loop=False, noAudio=True, size=(stim_width, stim_height), pos=(0, 0))

        movie_clip.play()
        movie_clip.draw()
        self.window.flip()
        

        while movie_clip.isFinished == False:
            movie_clip.play()
            movie_clip.draw()
            self.window.flip()
            self.ttl_clock.update()
            # core.wait(1)  # Freeze the video for a moment

        # Flush any keys in buffer
        event.clearEvents()

        # Display question
        options = trial['options'].split(',')
        question = trial['question']
        question += f"\n\n\n{self.corr_key[0]}. {options[0]} \t\t\t{self.corr_key[1]}. {options[1]}"
        question_stim = visual.TextStim(self.window, text=question, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=25)
        question_stim.draw()
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])
        trial['correct'] = (trial['response'] == options.index(str(trial['answer']))+1)

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        # Flush memory
        movie_clip.unload()
        gc.collect() # Collect garbarge

        return trial

class Movie(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.name = 'movie'

    def display_instructions(self):
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()

        self.instruction_text = f"\n\n You will watch short clips from a movie. Please keep your head still and pay attention to the screen."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 0))
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        window_width, _ = self.window.size
        movie_scale = self.const.movie_scale if hasattr(self.const, 'movie_scale') else 0.4
        stim_width = int(window_width * movie_scale) # Make the video fraction of the window width
        stim_height = int(stim_width  * 360 / 640)  # Original size of the video is 640x360
        
        # Get the file name
        movie_file_name = trial['stim']

        # Construct the movie file path
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / movie_file_name

        # Convert Path object to string for compatibility
        movie_path_str = str(movie_path)

        # Create a MovieStim3 object
        movie_clip = visual.MovieStim(self.window, movie_path_str, loop=False, size=(stim_width, stim_height), pos=(0, 0), noAudio=True)

        movie_clip.draw()
        movie_clip.play()
        self.window.flip()

        while movie_clip.isFinished == False:
            movie_clip.play()
            movie_clip.draw()
            self.window.flip()
            self.ttl_clock.update()

        # Flush memory
        movie_clip.unload()
        gc.collect() # Collect garbarge
        
        return trial
    

class StrangeStories(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.name = 'strange_stories'
    
    def init_task(self):
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0], self.trial_info['key_three'].iloc[0]]

    def display_instructions(self):
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 4))
        task_name.draw()

        self.instruction_text = f"\n\nYou will watch a clip about a couple and answer a question "
        if 'social' in self.task_file:
            self.instruction_text += "about their SOCIAL INTERACTION."
        elif 'control' in self.task_file:
            self.instruction_text += "about the FACTS."
        # self.instruction_text += " They live and work together. Each clip is self-contained and there is no story running from one clip to another."
        # self.instruction_text += "\n\n You will be asked a question about the clip. Imagine your answer as soon as you see the question. When you see the answer options, press the button that corresponds most to the answer you thought of. Some questions do not have a right or wrong answer."
        self.instruction_text += "\n\n Imagine your answer to the question. \nChoose the best match from the answers. \nSome questions have no right answer."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 0))
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        window_width, _ = self.window.size
        strange_stories_scale = self.const.strange_stories_scale if hasattr(self.const, 'strange_stories_scale') else 0.6
        stim_width = int(window_width * strange_stories_scale) # Make the video 40% of the window width
        stim_height = int(stim_width  * 921 / 1638)  # 1280x720 is the original size of the video given in width x height
        wrapWidth = 25
        
        # Get the file name
        movie_file_name = trial['stim']

        # Construct the movie file path
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / movie_file_name

        # Convert Path object to string for compatibility
        movie_path_str = str(movie_path)

        # Play the audio separately for better memory management
        play_audio_separatly = True

        # Create a MovieStim object
        movie_clip = visual.MovieStim(self.window, movie_path_str,
                                      loop=False, size=(stim_width, stim_height),
                                      pos=(0, 0), noAudio=play_audio_separatly)
        
        if play_audio_separatly:
            audio = self.get_audio_from_movie(movie_path, sample_rate=48000)
                
        movie_clip.draw()
        if play_audio_separatly:
            audio.play()
        movie_clip.play()
        self.window.flip()

        # Play through the movie frame by frame
        while movie_clip.isFinished == False:
            movie_clip.play()
            movie_clip.draw()
            self.window.flip()

        if play_audio_separatly:
            audio.stop()

        # Initialize question
        question = trial['question']

        # Initialize answer options
        options_orig = [answer_option.strip() for answer_option in trial['options'].split(',')]
        options_shuffled = deepcopy(options_orig)
        random.shuffle(options_shuffled) # Randomize the order of the answer options
        if 'control' in trial['condition']: # Only the first option is correct (2 points)
            scores_orig = [2,0,0]
        elif 'social'in trial['condition']: # First option gets 2 points, second option gets 1 point, third option gets 0 points
            scores_orig = [2,1,0]            
        scores_shuffled = [scores_orig[options_orig.index(option)] for option in options_shuffled]

        answers = f"\n\n\n{self.corr_key[0]}. {options_shuffled[0]} \n{self.corr_key[1]}. {options_shuffled[1]} \n{self.corr_key[2]}. {options_shuffled[2]}"

        # Display question
        stim_question = visual.TextStim(self.window, text = question, pos=(0, 4), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=wrapWidth)
        stim_question.draw()
        self.window.flip()
        # Display the question until X seconds before trial is over (answer_dur), to make the 'buffer' zone for the trial, i.e. the time of variable length, the time where the participant deliberates about their answer
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + (trial['trial_dur'] - movie_clip.duration - trial['answer_dur']))
        # Align the answers with the middle of the question if the answers are shorter than half of the question
        answer_lengths = [len(answer) for answer in options_shuffled]
        if max(answer_lengths) < wrapWidth and max(answer_lengths) < len(question):
            left_position = 0-max(answer_lengths)/3  # Answer options are shorter than questions and shorter than wrapWidth
            align='left'
        elif max(answer_lengths) >= wrapWidth:
            left_position = 0
            align='center'
        else:
            left_position = 0
            align='center'
        
        # Flush any keys in buffer
        event.clearEvents()
        stim_answers = visual.TextStim(self.window, text=answers, pos=(left_position, 0), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=wrapWidth, alignHoriz=align)
        stim_question.draw()
        stim_answers.draw()
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['answer_dur'])
        trial['acc'] = scores_shuffled[trial['response']-1]

        # Flush memory
        movie_clip.unload()
        gc.collect() # Collect garbarge

        return trial
    

class FauxPas(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def init_task(self):
        """
        Initialize task - default is to read the target information into the trial_info dataframe
        """
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['key_yes'].iloc[0],self.trial_info['key_no'].iloc[0]]

        
    def display_instructions(self):
        """
        displays the instruction for the task
        """
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()
        self.instruction_text = "\n\nRead the story and answer the Yes/No question "
        if 'social' in self.task_file:
            self.instruction_text += "about the SOCIAL INTERACTION."
        elif 'control' in self.task_file:
            self.instruction_text += "about the FACTS."
        self.instruction_text += f"\n\n{self.corr_key[0]}. Yes \t{self.corr_key[1]}. No\n"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 0))
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        """ Runs a single trial of the Theory of Mind task """

        event.clearEvents()

        # Display story
        story_stim = visual.TextStim(self.window, text=trial['story'], alignHoriz='center', wrapWidth=20, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height= 1.25)
        story_stim.draw()
        self.window.flip()

        # wait until story duration
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['story_dur'])

        # Flush any keys in buffer
        event.clearEvents()

        # Display question
        question = trial['question']
        # Display answers
        options = [option.strip(' ') for option in trial['options'].split(',')]
        question += f"\n\n\n{self.corr_key[0]}. {options[0]} \t\t\t{self.corr_key[1]}. {options[1]}"
        question_stim = visual.TextStim(self.window, text=question, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=25)
        question_stim.draw()
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(),
                                                           trial['question_dur'],
                                                           show_last_seconds=3,
                                                           current_stimuli=question_stim)
        trial['correct'] = (trial['response'] == trial['trial_type'])


        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        return trial
    

class FrithHappe(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.name = 'frith_happe'
        self.feedback_type = 'acc+rt'

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0], self.trial_info['key_three'].iloc[0]]

    def display_instructions(self):
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 4))
        task_name.draw()

        self.instruction_text = f"Decide how the two triangles are interacting."
        instr_stim = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 3))
        instr_stim.draw()
        answer_expalantion = f"\n\n{self.corr_key[0]}. No interaction\n\n{self.corr_key[1]}. Physical (The actions are directed towards each other) \n\n{self.corr_key[2]}. Mental (One triangle manipulates the thoughts or feelings of the other)"
        instr_visual = visual.TextStim(self.window, text=answer_expalantion, color=[-1, -1, -1], wrapWidth=20, pos=(-8, -1), alignHoriz='left')
        instr_visual.draw()
        self.window.flip()

    def run_trial(self, trial):
        window_width, _ = self.window.size
        frith_happe_scale = self.const.frith_happe_scale if hasattr(self.const, 'frith_happe_scale') else 0.4
        stim_width = int(window_width * frith_happe_scale) 
        stim_height = int(stim_width  * 1074 / 1433)
        wrapWidth = 25
        
        # Get the file name
        movie_file_name = trial['stim']
        # Construct the movie file path
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / movie_file_name
        # Convert Path object to string for compatibility
        movie_path_str = str(movie_path)
        # Create a MovieStim object
        movie_clip = visual.MovieStim(self.window, movie_path_str, loop=False, size=(stim_width, stim_height), pos=(0, 0), noAudio=True)
        
        movie_clip.draw()
        movie_clip.play()
        self.window.flip()

        # Play through the movie frame by frame
        while movie_clip.isFinished == False:
            movie_clip.play()
            movie_clip.draw()
            self.window.flip()
            self.ttl_clock.update()

        # Initialize question
        question = "What type of interaction did you see?"
        # Display question
        stim_question = visual.TextStim(self.window, text = question, pos=(0, 2), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=wrapWidth)
        stim_question.draw()
        self.window.flip()

        # Display the question until X seconds before trial is over (answer_dur), to make the 'buffer' zone for the trial, i.e. the time of variable length, the time where the participant deliberates about their answer
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + (trial['trial_dur'] - movie_clip.duration - trial['question_dur']))

        stim_question.draw()
        # Initialize answer options
        answers = f"\n\n{self.corr_key[0]}. No interaction \n{self.corr_key[1]}. Physical \n{self.corr_key[2]}. Mental"
        answers_stim = visual.TextStim(self.window, text=answers, pos=(-5, 0), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=wrapWidth, alignHoriz='left')
        answers_stim.draw()
        self.window.flip()

        # Flush any keys in buffer
        event.clearEvents()
        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])
        trial['correct'] = (trial['response'] == trial['trial_type'])

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        # Flush movie from memory
        movie_clip.unload()
        gc.collect() # Collect garbarge

        return trial
    


class Liking(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.name = 'liking'
        self.feedback_type = 'rt'

    def init_task(self):
        self.trial_info = pd.read_csv(self.const.task_dir / self.name / self.task_file, sep='\t')
        self.corr_key = [self.trial_info['key_one'].iloc[0],self.trial_info['key_two'].iloc[0], self.trial_info['key_three'].iloc[0], self.trial_info['key_four'].iloc[0]]

    def display_instructions(self):
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', color=[-1, -1, -1], bold=True, pos=(0, 5))
        task_name.draw()

        self.instruction_text = f"You will watch two people meeting for the first time."
        self.instruction_text += "\nRate how much they like each other."
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, color=[-1, -1, -1], wrapWidth=20, pos=(0, 2))
        instr_visual.draw()

        key_text = f"\n\n\n{self.corr_key[0]}. Dislike \n{self.corr_key[1]}. Mildly dislike \n{self.corr_key[2]}. Mildly like \n{self.corr_key[3]}. Like"
        # key_text = f"\n\n\n{self.corr_key[0]}. Not at all \n{self.corr_key[1]}. A little \n{self.corr_key[2]}. Moderately \n{self.corr_key[3]}. A lot"
        key_text = visual.TextStim(self.window, text=key_text, color=[-1, -1, -1], wrapWidth=20, pos=(-4, -1), alignHoriz='left')
        key_text.draw()
        self.window.flip()

    def run_trial(self, trial):
        window_width, _ = self.window.size
        liking_scale = self.const.liking_scale if hasattr(self.const, 'liking_scale') else 0.5
        stim_width = int(window_width * liking_scale)
        stim_height = int(stim_width  * 486 / 720) 
        wrapWidth = 20

        # Get the file name
        movie_file_name = trial['stim']
        # Construct the movie file path
        movie_path = Path(self.const.stim_dir) / self.name / 'clips' / (movie_file_name + '.mov')
        # Convert Path object to string for compatibility
        movie_path_str = str(movie_path)

        play_audio_separatly = True
        if play_audio_separatly:
            # Play the audio from the movie
            audio = self.get_audio_from_movie(movie_path, sample_rate=48000)

        # Create a MovieStim object
        movie_clip = visual.MovieStim(self.window, movie_path_str, loop=False,
                                    size=(stim_width, stim_height),
                                    pos=(0, 0),noAudio=play_audio_separatly)

        # Play through the movie frame by frame
        max_video_duration = 24
        movie_start_time = self.ttl_clock.get_time()
        
        movie_clip.draw()
        if play_audio_separatly:
            audio.play()
        movie_clip.play()
        self.window.flip()

        while self.ttl_clock.get_time() - movie_start_time < max_video_duration:
            movie_clip.play()
            movie_clip.draw()
            self.window.flip()
            self.ttl_clock.update()

        if play_audio_separatly:
            audio.stop()

        # Flush any keys in buffer
        event.clearEvents()

        # Initialize question
        question = "How much do they like each other?"
        # Display question
        stim_question = visual.TextStim(self.window, text = question, pos=(0, 3), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=wrapWidth)
        stim_question.draw()

        # Initialize answer options
        answers = f"\n\n{self.corr_key[0]}. Strongly dislike \n{self.corr_key[1]}. Dislike \n{self.corr_key[2]}. Like \n{self.corr_key[3]}. Strongly like"
        stim_answers = visual.TextStim(self.window, text=answers, pos=(-5, 0), color=(-1, -1, -1), units='deg', height= 1.25, wrapWidth=wrapWidth, alignHoriz='left')
        stim_answers.draw()
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])
        if trial['condition'] == 'like':
            trial['correct'] = (trial['response'] in [3, 4])
        elif trial['condition'] == 'dislike':
            trial['correct'] = (trial['response'] in [1, 2])
        else:
            trial['correct'] = False
        
        # Record the played video duration
        trial['video_dur_orig'] = trial['video_dur']
        trial['video_dur'] = max_video_duration
        
        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])

        # Flush memory
        movie_clip.unload()
        gc.collect() # Collect garbarge
        
        return trial
    
