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


class RestSurprise(Task):
    def __init__(self, info, screen, ttl_clock, const, subj_id):
        
        super().__init__(info, screen, ttl_clock, const, subj_id)

        self.name = 'rest_surprise'

        trial_info_file = (self.const.task_dir /self.name /self.task_file)
        self.trial_info = pd.read_csv(trial_info_file,sep='\t')

        self.trials = []

        grouped = self.trial_info.groupby('trial_num')

        for trial_num, group in grouped:

            group = group.sort_values('surprise_onset')

            # Convert rows into dictionaries
            trial_events = group.to_dict('records')

            self.trials.append(trial_events)

        self.red_flash = visual.Circle(
            win=self.window,
            radius=2,
            fillColor='red',
            lineColor='red',
            units='deg'
        )

        self.blue_flash = visual.Circle(
            win=self.window,
            radius=2,
            fillColor='blue',
            lineColor='blue',
            units='deg'
        )

        #self.low_beep = sound.Sound(
       # value=400,
       # secs=0.5
       # )

        #self.high_beep = sound.Sound(
        ##value=800,
        #secs=0.5
       # )
        
    def display_instructions(self): # overriding the display instruction routine from the parent
        self.instruction_text = 'Rest: Fixate on the cross'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text, height=self.const.instruction_text_height, color=[-1, -1, -1])
        # instr.size = 0.8
        instr_visual.draw()
        self.window.flip()
    
    def run(self):

        for trial_events in self.trials:

            self.run_trial(trial_events)
        return None, None

    def run_trial(self, trial_events):

        # Trial timing
        trial_start = self.ttl_clock.get_time()

        trial_duration = max(
            float(event['end_time'])
            for event in trial_events
        )

        # Track current event
        current_event_idx = 0

        # Visual stimulus state
        active_flash = None
        flash_end_time = None

        # -------------------------------------------------
        # Main trial loop
        # -------------------------------------------------
        while (
            self.ttl_clock.get_time() - trial_start
            < trial_duration
        ):

            # Elapsed trial time
            elapsed = (
                self.ttl_clock.get_time() - trial_start
            )

            # Draw fixation cross
            self.screen.fixation_cross()

            while (current_event_idx < len(trial_events) and elapsed >= float(trial_events[current_event_idx]['surprise_onset'])):
                    event = trial_events[current_event_idx]
                    stim_type = event['stimulus_type']
                    duration = float(event['duration'])

                    # -----------------------------------------
                    # AUDIO STIMULUS
                    # -----------------------------------------
                    if stim_type in ['audio', 'audiovisual']:

                        freq = event['freq']

                        if pd.notna(freq):
                            freq = float(freq)

                        #if freq == 400:
                         #   self.low_beep.play()

                        #elif freq == 800:
                         #   self.high_beep.play()

                    # -----------------------------------------
                    # VISUAL STIMULUS
                    # -----------------------------------------
                    if stim_type in ['visual', 'audiovisual']:

                        color = event['color']

                        if pd.notna(color):

                            if color == 'red':
                                active_flash = self.red_flash

                            elif color == 'blue':
                                active_flash = self.blue_flash

                        else:
                            active_flash = None

                        flash_end_time = (
                            elapsed + duration
                        )

                    # Move to next event
                    current_event_idx += 1

            # -------------------------------------------------
            # Draw active flash
            # -------------------------------------------------
            if active_flash is not None:

                if elapsed < flash_end_time:

                    active_flash.draw()

                else:

                    active_flash = None

            # Update display
            self.window.flip()

            core.wait(0.001)

            # Check for quit key
            self.screen_quit()

        return trial_events

class RestSurpriseImages(Task):

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.name = 'rest_surprise_images'

        trial_info_file = self.const.task_dir / self.name / self.task_file
        self.trial_info = pd.read_csv(trial_info_file, sep='\t')

        self.trials = []
        grouped = self.trial_info.groupby('trial_num')

        for trial_num, group in grouped:
            group = group.sort_values('surprise_onset')
            trial_events = group.to_dict('records')
            for evt in trial_events:
                stim_path = self.const.stim_dir / 'affective' / evt['stim']
                evt['_image'] = visual.ImageStim(self.window, str(stim_path))
            self.trials.append(trial_events)

    def display_instructions(self):
        self.instruction_text = 'Rest: Fixate on the cross'
        instr_visual = visual.TextStim(self.window, text=self.instruction_text,
                                       height=self.const.instruction_text_height, color=[-1, -1, -1])
        instr_visual.draw()
        self.window.flip()

    def run(self):
        for trial_events in self.trials:
            self.run_trial(trial_events)
        self.trial_data = pd.DataFrame()
        return None, None

    def run_trial(self, trial_events):
        trial_start = self.ttl_clock.get_time()
        trial_duration = max(float(evt['end_time']) for evt in trial_events)

        current_event_idx = 0
        active_image = None
        image_end_time = None

        while self.ttl_clock.get_time() - trial_start < trial_duration:
            elapsed = self.ttl_clock.get_time() - trial_start

            while (current_event_idx < len(trial_events) and
                   elapsed >= float(trial_events[current_event_idx]['surprise_onset'])):
                evt = trial_events[current_event_idx]
                active_image = evt['_image']
                image_end_time = elapsed + float(evt['duration'])
                current_event_idx += 1

            if active_image is not None and elapsed < image_end_time:
                active_image.draw()
            else:
                if active_image is not None:
                    active_image = None
                self.screen.fixation_cross(flip=False)

            self.window.flip()
            core.wait(0.001)
            self.screen_quit()

        return trial_events

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

class RestSurprise(TaskFile):

    def __init__(self, const):
        super().__init__(const)
        self.name = 'rest_surprise'

    def make_task_file(
        self,
        task_dur=30,
        min_interval=3,
        max_interval=8,
        stim_dur=0.5,
        file_name=None
    ):

        trial_info = []

        t = random.uniform(min_interval, max_interval)
        event_num = 0

        while t < task_dur:

            trial = {}

            surprise_type = random.choice([
                'audio',
                'visual',
                'audiovisual'
            ])

            if surprise_type == 'audio':
                freq = random.choice([400, 800])
                color = "None"

            elif surprise_type == 'visual':
                color = random.choice([
                    'red',
                    'blue'
                ])
                freq = "None" 

            elif surprise_type == 'audiovisual':
                color = random.choice([
                    'red',
                    'blue'
                ])
                freq = random.choice([400,800])

            trial['trial_num']=1
            trial['event_num']=event_num
            trial['surprise_onset']=round(t, 2)
            trial['duration']=stim_dur
            trial['surprise_end'] = round(t + stim_dur, 2)   

            trial['stimulus_type']=surprise_type
            trial['color']=color
            trial['freq']=freq 
            trial['start_time']=0
            trial['end_time']=task_dur
            trial_info.append(trial)

            t += random.uniform(min_interval, max_interval)
            event_num += 1

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name,sep='\t',index=False)


        return trial_info

class RestSurpriseImages(TaskFile):

    def __init__(self, const):
        super().__init__(const)
        self.name = 'rest_surprise_images'

    def make_task_file(
        self,
        task_dur=30,
        n_stimuli=4,
        stim_dur=0.5,
        file_name=None
    ):
        pleasant_imgs = [f'pleasant{i}.jpg' for i in range(1, 27)]
        unpleasant_imgs = [f'unpleasant{i}.jpg' for i in range(1, 55)]
        all_imgs = pleasant_imgs + unpleasant_imgs

        onsets = sorted([random.uniform(0, task_dur - stim_dur) for _ in range(n_stimuli)])

        trial_info = []
        for event_num, t in enumerate(onsets):
            stim_file = random.choice(all_imgs)
            category = 'pleasant' if stim_file.startswith('pleasant') else 'unpleasant'
            trial = {}
            trial['trial_num'] = 1
            trial['event_num'] = event_num
            trial['surprise_onset'] = round(t, 2)
            trial['duration'] = stim_dur
            trial['surprise_end'] = round(t + stim_dur, 2)
            trial['stimulus_type'] = 'visual'
            trial['stim'] = stim_file
            trial['category'] = category
            trial['start_time'] = 0
            trial['end_time'] = task_dur
            trial_info.append(trial)

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class RestSurpriseSoundImages(TaskFile):

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
