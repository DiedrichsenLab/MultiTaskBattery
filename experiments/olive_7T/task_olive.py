""" define the tasks unique to the Olive7T experiment"""

from MultiTaskBattery.task_blocks import Task
from MultiTaskBattery.task_file import TaskFile
from pathlib import Path
import pandas as pd
import numpy as np
import random
import importlib.metadata as _importlib_metadata
_orig_entry_points = _importlib_metadata.entry_points
def _patched_entry_points(group=None, **kwargs):
    result = _orig_entry_points(**kwargs)
    if group is not None:
        return result.get(group, [])
    return result
_importlib_metadata.entry_points = _patched_entry_points
from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']
from psychopy import visual, sound, core, event
from pyglet.window import key
import MultiTaskBattery.utils as ut
from ast import literal_eval
from copy import deepcopy
from moviepy.audio.io.AudioFileClip import AudioFileClip
import gc
import math
import json
import soundfile as sf
import sounddevice as sd

class RestSurpriseSoundImages(Task):

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.name = 'rest_surprise_sound_images'

        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')

        self.trial_events = self.trial_info.sort_values('surprise_onset').to_dict('records')
        for evt in self.trial_events:
            stim_type = evt['stimulus_type']
            if stim_type in ['visual', 'audiovisual'] and evt['stim']:
                img_path = self.const.stim_dir / 'affective' / evt['stim']
                evt['_image'] = visual.ImageStim(self.window, str(img_path))
            else:
                evt['_image'] = None
            if stim_type in ['auditory', 'audiovisual'] and evt['sound_stim']:
                snd_path = self.const.stim_dir / evt['sound_dir'] / evt['sound_stim']
                data, sr = sf.read(str(snd_path))
                evt['_sound_data'] = data
                evt['_sound_sr'] = sr
            else:
                evt['_sound_data'] = None

    def display_instructions(self):
        self.instruction_text = 'Rest: Fixate on the cross'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text,
                                       height=self.const.instruction_text_height, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run(self):
        self.run_trial(self.trial_events)
        self.trial_data = pd.DataFrame()
        return None, None

    def run_trial(self, trial_events):
        trial_start = self.ttl_clock.get_time()
        trial_duration = max(float(evt['end_time']) for evt in trial_events)

        current_event_idx = 0
        active_image = None
        image_end_time = None
        active_sound = False
        sound_end_time = None

        while self.ttl_clock.get_time() - trial_start < trial_duration:
            elapsed = self.ttl_clock.get_time() - trial_start

            while (current_event_idx < len(trial_events) and
                   elapsed >= float(trial_events[current_event_idx]['surprise_onset'])):
                evt = trial_events[current_event_idx]
                stim_type = evt['stimulus_type']

                if stim_type in ['visual', 'audiovisual'] and evt['_image'] is not None:
                    active_image = evt['_image']
                    image_end_time = elapsed + float(evt['duration'])

                if stim_type in ['auditory', 'audiovisual'] and evt.get('_sound_data') is not None:
                    sd.stop()
                    sd.play(evt['_sound_data'], evt['_sound_sr'])
                    active_sound = True
                    sound_end_time = elapsed + float(evt['duration'])
                current_event_idx += 1

            if active_image is not None and elapsed >= image_end_time:
                active_image = None

            if active_sound and elapsed >= sound_end_time:
                sd.stop()
                active_sound = False

            if active_image is not None:
                active_image.draw()
            else:
                self.screen.fixation_cross(flip=False)

            self.window.flip()
            core.wait(0.001)
            self.screen_quit()

        if active_sound is not None:
            sd.stop()

        return trial_events

class RestSurpriseSoundImagesFile(TaskFile):

    def __init__(self, const):
        super().__init__(const)
        self.name = 'rest_surprise_sound_images'

    def make_task_file(
        self,
        task_dur=30,
        n_stimuli=4,
        img_stim_dur=0.5,
        sound_dir='degraded_passage',
        file_name=None
    ):
        pleasant_imgs = [f'pleasant{i}.jpg' for i in range(1, 27)]
        unpleasant_imgs = [f'unpleasant{i}.jpg' for i in range(1, 55)]
        all_imgs = pleasant_imgs + unpleasant_imgs

        sound_path = self.stim_dir / sound_dir
        sound_files = [f.name for f in sound_path.glob('*.wav')] + [f.name for f in sound_path.glob('*.mp3')]

        stim_types = ['visual', 'auditory', 'audiovisual']
        onsets = sorted([random.uniform(0, task_dur - img_stim_dur) for _ in range(n_stimuli)])

        trial_info = []
        for event_num, t in enumerate(onsets):
            stim_type = random.choice(stim_types)
            trial = {}
            trial['trial_num'] = 1
            trial['event_num'] = event_num
            trial['surprise_onset'] = round(t, 2)
            trial['stimulus_type'] = stim_type
            trial['duration'] = img_stim_dur
            trial['start_time'] = 0
            trial['end_time'] = task_dur

            if stim_type in ['visual', 'audiovisual']:
                img = random.choice(all_imgs)
                trial['stim'] = img
                trial['category'] = 'pleasant' if img.startswith('pleasant') else 'unpleasant'
            else:
                trial['stim'] = ''
                trial['category'] = ''

            if stim_type in ['auditory', 'audiovisual']:
                trial['sound_stim'] = random.choice(sound_files)
                trial['sound_dir'] = sound_dir
            else:
                trial['sound_stim'] = ''
                trial['sound_dir'] = ''

            trial_info.append(trial)

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info
    
class FingerSequenceSurprise(Task):
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
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, height=self.const.instruction_text_height, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()


    def run_trial(self, trial):
        """ Run a single trial of the finger sequence surprise task. """
        #clear buffer
        event.clearEvents()

        real_start_time, start_ttl, start_ttl_time = self.ttl_clock.wait_until(
        trial['start_time']
    )

        # Display the sequence
        original_sequence = trial['stim_original'].split()
        final_sequence = trial['stim_final'].split()

        displayed_sequence = original_sequence.copy()

        num_items = len(displayed_sequence)

        # Calculate the start position for the sequence and determine the spacing between numbers
        spacing = 2.0
        start_x = -(num_items - 1) * spacing / 2

        has_change = pd.notna(trial['change_type'])

        if has_change:
            change_position = int(trial['position_change']) - 1
            shift = 1 if trial['change_type'] == 'first_horizon' else 2
            trigger_press = change_position - shift
        else:
            trigger_press = None


        rt_list = np.full(num_items, np.nan)
        correct_list = np.zeros(num_items)
        num_presses = 0
        sequence_start_time = self.ttl_clock.get_time()
        digit_start_time = sequence_start_time
        change_applied = False

        stim_objects = []
        
        for i, number in enumerate(displayed_sequence):
            pos = (start_x + i * spacing, 0.0)  # Horizontal position is adjusted based on index
            stim = visual.TextStim(self.window, text=number, pos=pos, color='black', units='deg', height=1.5)
            stim_objects.append(stim)

        while (
            self.ttl_clock.get_time() - sequence_start_time < trial['trial_dur']) and (num_presses < num_items):
            self.ttl_clock.update()
            elapsed = self.ttl_clock.get_time() - sequence_start_time

            if (
                has_change
                and not change_applied
                and num_presses > trigger_press
              ):
                
                displayed_sequence[change_position] = final_sequence[change_position]

                stim_objects[change_position].text = final_sequence[change_position]
                
                change_applied = True

            keys = event.getKeys(
                keyList=self.const.response_keys,
            timeStamped=self.ttl_clock.clock
        )
            
            if keys:
                
                key_char, key_press_time = keys[0]

                response = self.const.response_keys.index(key_char) + 1
                
                rt = key_press_time - digit_start_time
                
                rt_list[num_presses] = rt
                
                digit_start_time = key_press_time

                correct_digit = int(displayed_sequence[num_presses])
                
                is_correct = response == correct_digit
                
                correct_list[num_presses] = is_correct
                
                stim_objects[num_presses].color = (
                    'green' if is_correct else 'red'
                    )
                
                num_presses += 1

        # Show the numbers in the sequence next to each other ( using the spacing and start_x calculated above)

            for stim in stim_objects:
                stim.draw()

            self.window.flip()

        self.ttl_clock.wait_until(sequence_start_time + trial['trial_dur'])

        trial['correct'] = correct_list.sum() / num_items

        if np.all(np.isnan(rt_list)):
            trial['rt'] = np.nan
        else:
            trial['rt'] = np.nanmean(rt_list)

        # display trial feedback (for whole trial)
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct']== 1)

        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl
        trial['start_ttl_time'] = start_ttl_time

        return trial
    
class FingerSequenceSurpriseFile(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'finger_sequence_surprise'
        self.matching_stimuli = False # sequence of numbers are different for easy and hard sequence condition

    def generate_sequence(self):
        sequence = [random.choice([1, 2, 3, 4])]
        while len(sequence) < 6:
            next_digit = random.choice([d for d in [1, 2, 3, 4] if d != sequence[-1]])
            sequence.append(next_digit)
        return sequence

    def make_task_file(self,
                        hand = 'unimanual',
                        responses = [1,2,3,4], # 1 = Key_one, 2 = Key_two, 3 = Key_three, 4 = Key_four
                        task_dur=30,
                        trial_dur=3.25,
                        iti_dur=0.5,
                        file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        # randomly choose 50% of trials to contain changes
        change_trials = random.sample(
            range(n_trials),
            k=n_trials // 2) 

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['key_one'] = responses[0]
            trial['key_two'] = responses[1]
            trial['key_three'] = responses[2]
            trial['key_four'] = responses[3]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = True
            # choose random sequence
            sequence = self.generate_sequence()
            trial['change_type']="None"
            trial['change_time']="None"
            trial['position_change']="None"
            trial['change_to']= "None"
            trial['stim_original']=' '.join(map(str, sequence))

            if n in change_trials:
                shift = random.choice([1,2])
                base_position = random.choice(range(0,6-shift))
                change_position = base_position + shift
                original_digit = sequence[change_position]
                possible_digits = [1,2,3,4] 
                possible_digits.remove(original_digit)  

                if change_position > 0: 
                    prev_digit = sequence[change_position - 1]  
                    if prev_digit in possible_digits:
                        possible_digits.remove(prev_digit)

                if change_position < 5:
                    next_digit = sequence[change_position + 1]  
                    if next_digit in possible_digits:
                        possible_digits.remove(next_digit)

                new_digit = random.choice(possible_digits)  

                sequence[change_position] = new_digit

                digit_time = trial_dur / 6 

                change_time = t+((base_position+1)*digit_time)  

                trial['change_type']='first_horizon' if shift ==1 else "second_horizon"
                trial['change_time'] = round(change_time, 2)
                trial['position_change'] = change_position + 1 
                trial['change_to'] = new_digit
            
            trial['stim_final'] = ' '.join(map(str, sequence))

            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info
    
class TheoryOfMindDiffReward(Task):
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
        task_name = visual.TextStim(self.window, text=f'{self.descriptive_name.capitalize()}', height=self.const.instruction_text_height, color=[-1, -1, -1], bold=True, pos=(0, 3))
        task_name.draw()
        str1 = f"You will read a story and decide if the answer to the question is True or False. Before each story, you will see how many points the trial is worth."
        str2 = f"if true, press {self.corr_key[1]}"
        str3 = f"if false, press {self.corr_key[0]}"
        self.instruction_text = f"\n\n\n {str1} \n\n {str2} \n {str3}"
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, height=self.const.instruction_text_height, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def display_trial_feedback(self, give_feedback, correct_response, reward_cue=None):
        if give_feedback:
            if correct_response:
                text = f"+{reward_cue}"
                color = 'green'
            else:
                text = "-1"
                color = 'red'
            feedback_stim = visual.TextStim(
                self.window,
                text=text,
                color=color,
                height=2.0,
                bold=True
        )
            feedback_stim.draw()
            self.window.flip()
        else:
            self.screen.fixation_cross('white')

    def run_trial(self, trial):
        """ Runs a single trial of the Theory of Mind task """

        real_start_time, start_ttl, start_ttl_time = (
            self.ttl_clock.wait_until(
            trial['start_time']
            )
        )

        event.clearEvents()

        height = trial.get('text_height', 1.25)
        wrapWidth=25

        reward_text = trial['reward_cue']

        reward_stim = visual.TextStim(
            self.window,
            text=reward_text,
            height=2.0,
            color='yellow',
            bold=True,
            pos=(0, 0)
        )

        reward_stim.draw()

        self.window.flip()

        reward_start = self.ttl_clock.get_time()

        self.ttl_clock.wait_until(
            reward_start
            + trial['reward_cue_dur']
        )

        story_clean = ' '.join(trial['story'].split('\n'))
        story_formatted = '.\n'.join(story_clean.split('. '))
        story_stim = visual.TextStim(self.window, text=story_formatted, alignHoriz='center', wrapWidth=wrapWidth, pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=height)
        story_stim.draw()
        self.window.flip()

        # wait until story duration

        story_start = self.ttl_clock.get_time()

        self.ttl_clock.wait_until(
            story_start + trial['story_dur']
        )

        # Flush any keys in buffer
        event.clearEvents()

        # Display question
        question_stim = visual.TextStim(self.window, text=trial['question'], pos=(0.0, 0.0), color=(-1, -1, -1), units='deg', height=height, wrapWidth=25)
        question_stim.draw()
        self.window.flip()

        # collect responses 0: no response 1-4: key pressed
        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['question_dur'])
        trial['correct'] = (trial['response'] == self.corr_key[trial['trial_type']])

        if trial['correct']:

            trial['points_earned'] = (
                trial['reward_value']
            )

        else:

            trial['points_earned'] = 0

        # display trial feedback
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'], trial['reward_cue'])
        give_feedback = trial['display_trial_feedback'] or (trial['response'] == 0)
        self.display_trial_feedback(give_feedback, trial['correct'], trial['reward_cue'])

        if trial['response'] == 0:
            self.ttl_clock.wait_until(self.ttl_clock.get_time() + 1.0)

        trial['real_start_time'] = (
            real_start_time
        )

        trial['start_ttl'] = start_ttl

        trial['start_ttl_time'] = (
            start_ttl_time
        )

        return trial
    
class TheoryOfMindDiffRewardFile(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'theory_of_mind_diff_reward'
        self.matching_stimuli = False # stimuli for active condition (belief) are different from stimuli for passive condition (photo)

    def make_task_file(self, hand='right',
                       responses = [1,2], # 1 = True, 2 = False
                       run_number=None,
                        task_dur=30,
                        trial_dur=15,
                        iti_dur=0,
                        reward_cue_dur=1,
                        story_dur=10,
                        question_dur=4,
                        text_height=1.25,
                        file_name=None,
                        stim_file=None,
                        condition=None,
                        stimulus_seed=None,
                        exclude_stimuli=None,
                        stim=None):
        # Count number of trials based on timing; may be overridden below when an
        # explicit stimulus list is provided (so distribution, not timing, sets
        # the exact trial count).
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        high_reward_trials = random.sample(
            range(n_trials),
            k=n_trials // 2)

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / 'theory_of_mind' / 'theory_of_mind.csv')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[
                ~stim['condition'].str.contains('practice', na=False)
                & (stim['condition'].astype(str).str.lower() != 'exclude')
            ]

        # Ignore stim_list and stimulus_seed: selection is driven entirely by
        # the provided stim_file (if any) or by run_number-based slicing.
        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        n_actual = min(n_trials, len(stim))
        for n in range(n_actual):
            trial = {}
            trial['key_true'] = responses[0]
            trial['key_false'] = responses[1]
            if str(stim['answer'][n]) == 'True':
                trial['trial_type'] = 1
            else:
                trial['trial_type'] = 0
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['story'] = stim['story'][n]
            trial['question'] = stim['question'][n]
            trial['condition'] = stim['condition'][n]
            trial['answer'] = stim['answer'][n]
            trial['story_dur'] = story_dur
            trial['question_dur'] = question_dur
            trial['text_height'] = text_height

            if n in high_reward_trials:
                trial['reward_value'] = 3
                trial['reward_cue'] = '+3'
            else:
                trial['reward_value'] = 1
                trial['reward_cue'] = '+1'

            trial['reward_cue_dur'] = reward_cue_dur    
            trial['display_trial_feedback'] = True
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return trial_info
    
class TempDeviant(Task):
    """
    Subjects see a flashing disk and must indicate
    how many temporal deviants occurred (0,1,2,3 or 4)
    """

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc'

    def init_task(self):

        trial_info_file = (
            self.const.task_dir
            / self.name
            / self.task_file
        )

        self.trial_info = pd.read_csv(
            trial_info_file,
            sep='\t'
        )

        self.response_options = [0, 1, 2, 3]

    def display_instructions(self):

        self.instruction_text = (
            f"{self.descriptive_name} Task\n\n"
            "Watch the flashing disk.\n\n"
            "Indicate if there are deviant stimuli."
        )

        instr_visual = visual.TextStim(
            self.window,
            text=self.instruction_text,
            height=self.const.instruction_text_height,
            color=[-1, -1, -1]
        )

        instr_visual.draw()

        self.window.flip()

    def run_trial(self, trial):

        event.clearEvents()

        real_start_time, start_ttl, start_ttl_time = (
            self.ttl_clock.wait_until(
                trial['start_time']
            )
        )

        isi = float(trial['stim_interval'])

        sequence_duration = float(
            trial['trial_duration']
        )

        response_duration = float(
            trial['response_duration']
        )

        # Expected flash times
        flash_times = np.arange(
            0,
            sequence_duration,
            isi
        )

        modified_times = flash_times.copy()

        if pd.notna(trial['deviant_positions']):

            dev_positions = [
                int(float(x)) - 1
                for x in str(
                    trial['deviant_positions']
                ).split(';')
            ]

            dev_times = [
                float(x)
                for x in str(
                    trial['deviant_times']
                ).split(';')
            ]

            relative_dev_times = [
                t - trial['start_time']
                for t in dev_times
            ]

            for pos, dev_time in zip(
                dev_positions,
                relative_dev_times
            ):

                modified_times[pos] = dev_time

        disk = visual.Circle(
            self.window,
            radius=2.0,
            fillColor='white',
            lineColor='white',
            units='deg'
        )

        fixation = visual.TextStim(
            self.window,
            text='+',
            color='white',
            height=1.0
        )

        sequence_start = self.ttl_clock.get_time()

        flash_duration = 0.1

        for flash_time in modified_times:

            # Wait until flash onset
            self.ttl_clock.wait_until(
                sequence_start + flash_time
            )

            # Draw flash
            disk.draw()

            self.window.flip()

            # Hold flash briefly
            self.ttl_clock.wait_until(
                sequence_start
                + flash_time
                + flash_duration
            )

            fixation.draw()

            self.window.flip()

        response_text = visual.TextStim(
            self.window,
            text=(
                "Were there any deviant stimuli?\n\n"
                "Yes - press 1\n"
                "No - press 2"
            ),
            color='white',
            height=0.8
        )

        response_text.draw()

        self.window.flip()

        trial['response'],trial['rt'] = self.wait_response(self.ttl_clock.get_time(), response_duration)
        trial['correct'] = (trial['response'] == (1 if trial['n_deviants'] > 0 else 2))
        
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])
        trial['real_start_time'] = real_start_time
        trial['start_ttl'] = start_ttl

        return trial
    
class TempDeviantFile(TaskFile):

    def __init__(self, const):
        super().__init__(const)
        self.name = 'temp_deviant'

    def make_task_file(
        self,
        task_dur=30,
        sequence_dur=13,
        response_dur=2,
        frequency=1,
        deviant_shift=0.2,
        display_trial_feedback=True,
        file_name=None
    ):
        
        trial_info = []

        trial_total_dur = sequence_dur + response_dur

        n_trials = int(np.floor(task_dur / trial_total_dur))

        t = 0

        # 50% no-deviant trials
        n_deviant_trials = n_trials // 2

        deviant_trials = random.sample(
            range(n_trials),
            k=n_deviant_trials
        )

        for n in range(n_trials):

            trial = {}

            isi = 1 / frequency

            # expected flash times
            flash_times = np.arange(
                0,
                sequence_dur,
                isi
            )

            trial['trial_num'] = n
            trial['frequency'] = frequency
            trial['stim_interval'] = isi
            trial['trial_duration'] = sequence_dur
            trial['response_duration'] = response_dur
            trial['display_trial_feedback'] = display_trial_feedback

            trial['start_time'] = t
            trial['end_time'] = t + trial_total_dur

            trial['n_deviants'] = 0
            trial['deviant_times'] = "None"
            trial['deviant_type'] = "None"
            trial["deviant_positions"] = "None"


            modified_times = flash_times.copy()

            if n in deviant_trials:

                # choose 1,2, or 3 deviants equally
                n_dev = random.choice([1, 2, 3])

                trial['n_deviants'] = n_dev

                # avoid first and last flashes
                possible_positions = list(
                    range(3, len(flash_times) - 1)
                )

                deviant_positions = sorted(
                    random.sample(
                        possible_positions,
                        k=n_dev
                    )
                )
                trial['deviant_positions'] = ';'.join(map(str, [p + 1 for p in deviant_positions]))

                deviant_times = []
                deviant_types = []

                for pos in deviant_positions:

                    shift_direction = random.choice(
                        [-1, 1]
                    )

                    shift_label = (
                        'early'
                        if shift_direction < 0
                        else 'late'
                    )

                    modified_times[pos] += (
                        shift_direction * deviant_shift
                    )

                    deviant_times.append(
                        round(
                            t + modified_times[pos],
                            2
                        )
                    )

                    deviant_types.append(
                        shift_label
                    )

                trial['deviant_times'] = (
                    ';'.join(map(str, deviant_times))
                )

                trial['deviant_type'] = (
                    ';'.join(deviant_types)
                )

            trial_info.append(trial)

            t += trial_total_dur

        trial_info = pd.DataFrame(trial_info)

        if file_name is not None:

            ut.dircheck(self.task_dir / self.name)

            trial_info.to_csv(
                self.task_dir / self.name / file_name,
                sep='\t',
                index=False
            )

        return trial_info
    

    

