# Create TaskFile file for different tasks
# March 2021: First version: Ladan Shahshahani  - Maedbh King - Suzanne Witt,
# Revised 2023: Bassel Arafat, Jorn Diedrichsen, Ince Husain

import pandas as pd
import numpy as np
import random
import MultiTaskBattery.utils as ut
import itertools


def shuffle_rows(dataframe, keep_in_middle=None):
    """
    randomly shuffles rows of the dataframe

    Args:
        dataframe (dataframe): dataframe to be shuffled
        keep_in_middle (list): list of tasks that should be kept in the middle
    Returns:
        dataframe (dataframe): shuffled dataframe
    """
    indx = np.arange(len(dataframe.index))
    np.random.shuffle(indx)
    dataframe = (dataframe.iloc[indx]).reset_index(drop = True)

    if keep_in_middle is not None:
        dataframe = move_edge_tasks_to_middle(dataframe, keep_in_middle)

    return dataframe

def move_edge_tasks_to_middle(dataframe, keep_in_middle):
    """Moves tasks that should be kept in the middle and are at the edge of the dataframe to the middle of the dataframe
    Args:
        dataframe (dataframe): dataframe to be shuffled
        keep_in_middle (list): list of tasks that should be kept in the middle
    Returns:
        dataframe (dataframe): shuffled dataframe"""

    middle_task_at_edge = np.any([edge_task in keep_in_middle for edge_task in dataframe.iloc[[0, -1]].task_name])

    if middle_task_at_edge and len(keep_in_middle) < len(dataframe)-2:
        # If the middle task is at the edge and there are enough tasks to frame the middle tasks, find a new position in the middle
        middle_indices = list(dataframe.iloc[1:-1].index)
        middle_indices = [i for i in middle_indices if dataframe.iloc[i].task_name not in keep_in_middle]
        selected_middle_indices = np.random.choice(middle_indices, len(keep_in_middle), replace=False)

        for i, task in enumerate(keep_in_middle):
            # Find where the middle task is wrongly placed (edge positions)
            for edge_idx in [0, -1]:
                if dataframe.iloc[edge_idx].task_name == task:
                    # Swap the edge task with a chosen middle position
                    middle_idx = selected_middle_indices[i]
                    dataframe.iloc[edge_idx], dataframe.iloc[middle_idx] = dataframe.iloc[middle_idx], dataframe.iloc[edge_idx]
    return dataframe


def add_start_end_times(dataframe, offset, task_dur, run_time=None):
    """
    adds start and end times to the dataframe

    Args:
        dataframe (dataframe): dataframe to be shuffled
        offset (float): offset of the task
        task_dur (float): duration of the task
        run_time (float): Time (in seconds) that the run should last. Use this to ensure the last task runs until the end of the imaging run
    Returns:
        dataframe (dataframe): dataframe with start and end times
    """
    dataframe['start_time'] = np.arange(offset, offset + len(dataframe)*task_dur, task_dur)
    dataframe['end_time']   = dataframe['start_time'] + task_dur
    if run_time:
        if run_time < dataframe['end_time'].iloc[-1]:
            raise ValueError('Run time is shorter than the last task')
        # Add add_end_time seconds to the last task to ensure the task runs until the end of the run (e.g. for capturing the activity overhang from the final task in an imaging run)
        dataframe.loc[dataframe.index[-1], 'end_time'] = run_time
    return dataframe

def make_run_file(task_list,
                  tfiles,
                  offset = 0,
                  instruction_dur = 5,
                  task_dur = 30,
                  run_time = None,
                  keep_in_middle=None,
                  exp_dir=None):
    """
    Make a single run file
    """
    task_table = ut.get_task_table(exp_dir)
    # Get rows of the task_table corresponding to the task_list
    indx = [np.where(task_table['name']==t)[0][0] for t in task_list]
    R = {'task_name':task_list,
         'task_code':task_table['code'].iloc[indx],
         'task_file':tfiles,
         'instruction_dur':[instruction_dur]*len(task_list),
         'task_dur':[task_dur]*len(task_list)}
    R = pd.DataFrame(R)
    R = shuffle_rows(R, keep_in_middle=keep_in_middle)
    R = add_start_end_times(R, offset, task_dur+instruction_dur, run_time=run_time)
    return R

def get_task_class(name, exp_dir=None):
    """Creates an object of the task class based on the task name
    Args:
        name (str): name of the task
        exp_dir (str, path, optional): path to the experiment directory
    Returns:
        class_name (str): class name for task
    """
    task_table = ut.get_task_table(exp_dir)
    index = np.where(task_table['name']==name)[0][0]
    class_name = task_table.iloc[index]['task_class']
    return class_name

class TaskFile():
    def __init__(self, const) :
        """ The TaskFile class is class for creating TaskFile files for different tasks
        Args:
            const: module for constants
        """
        self.exp_name           = const.exp_name
        self.task_dir           = const.task_dir
        self.stim_dir           = const.stim_dir
        self.matching_stimuli   = True # whether the stimuli are matching between the control and active condition or not
        self.half_assigned      = False # whether the stimuli have assigne halves or not (for assigning different stimuli to different participants)


class NBack(TaskFile): # with the 5 stimuli used here only 1-back/2-back/3-back is safe and tested
    def __init__(self, const):
        super().__init__(const)
        self.name = 'n_back'

    def make_task_file(self,
                        hand = 'right',
                        responses = [1,2], # 1 = match, 2 = no match
                        task_dur =  30,
                        trial_dur = 2,
                        iti_dur   = 0.5,
                        picture_scale = 1.0,
                        n_back = 2, # number of items back a match refers to (2 = classic 2-back)
                        stim = ['9.jpg','11.jpg','18.jpg','28.jpg'],
                        file_name = None ):
        """
        Create an n-back working-memory task file.

        Args:
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys for [match, no-match].
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration each stimulus is displayed in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            picture_scale (float): Scaling factor for stimulus images (>1 enlarges).
            n_back (int): How many items back a match refers to (2 = classic 2-back).
            stim (list): List of stimulus image filenames to draw from.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        prev_stim = ['x'] * n_back  # sliding window of the last n_back stimuli (index n_back-1 = the n-back item)
        t = 0
        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['picture_scale'] = picture_scale
            # The n-back level is recorded once, as the modeling-level condition
            # (e.g. '2-back'). n_back itself is only a generation parameter, so it
            # is not duplicated as its own column - the runtime derives n from here.
            trial['condition'] = f"{n_back}-back"
            trial['display_trial_feedback'] = True
            trial['key_match'] = responses[0]
            trial['key_nomatch'] = responses[1]
            # Determine if this should be an n-back repetition trial
            if n < n_back:
                trial['trial_type'] = 0
            else:
                trial['trial_type'] = np.random.randint(0,2)
            # Now choose the stimulus accordingly: avoid any reps within the window
            if trial['trial_type']==0:
                trial['stim'] = prev_stim[n_back-1]
                while trial['stim'] in prev_stim:
                    trial['stim'] = stim[np.random.randint(0,len(stim))]
            else:
                trial['stim'] = prev_stim[n_back-1]

            trial['display_trial_feedback'] = True
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial: slide the window, newest stimulus first
            t= trial['end_time']
            prev_stim = [trial['stim']] + prev_stim[:-1]

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name,sep='\t',index=False)
        return trial_info

class Rest(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'rest'

    def make_task_file(self,
                        task_dur =  30,
                        file_name = None):
        """
        Create a rest task file (single fixation block, no stimuli or response).

        Args:
            task_dur (float): Total duration of the rest block in seconds.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        trial = {}
        trial['trial_num'] = [0]
        trial['trial_dur'] = [task_dur]
        trial['start_time'] = [0]
        trial['end_time'] =  [task_dur]
        trial_info = pd.DataFrame(trial)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name,sep='\t',index=False)
        return trial_info
    

class VerbGeneration(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'verb_generation'


    def make_task_file(self,
                        condition = ['read', 'generate'],
                        task_dur =  30,
                        trial_dur = 2,
                        iti_dur   = 0.5,
                        order = 'blocked',
                        file_name = None,
                        stim_file = None):
        """
        Create a verb-generation task file.

        Args:
            condition (str or list): Which condition(s) to run. A single value
                ('read' or 'generate') fills the whole block with one condition.
                A list (e.g. ['read', 'generate']) mixes conditions within the
                block.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration each word is displayed in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            order (str): How to arrange multiple conditions across trials:
                'blocked' (default) runs each condition in a contiguous chunk in
                the given order; 'interleaved' cycles through them trial by trial;
                'random' assigns a balanced set in random order. Ignored for a
                single condition.
            file_name (str): Name of the file to save the task data.
            stim_file (str): Optional path to a custom word-list CSV. Defaults to
                the packaged verb_generation.csv.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        conditions = [condition] if isinstance(condition, str) else list(condition)
        for c in conditions:
            if c not in ('read', 'generate'):
                raise ValueError(f"VerbGeneration: condition must be 'read' or 'generate', got {c!r}")

        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))

        # Assign a condition to each trial.
        if len(conditions) == 1:
            trial_conditions = [conditions[0]] * n_trials
        elif order == 'interleaved':
            trial_conditions = [conditions[i % len(conditions)] for i in range(n_trials)]
        elif order == 'random':
            balanced = (conditions * (n_trials // len(conditions) + 1))[:n_trials]
            trial_conditions = [str(c) for c in np.random.permutation(balanced)]
        elif order == 'blocked':
            per = n_trials // len(conditions)
            trial_conditions = []
            for i, c in enumerate(conditions):
                count = per if i < len(conditions) - 1 else n_trials - per * (len(conditions) - 1)
                trial_conditions += [c] * count
        else:
            raise ValueError(f"VerbGeneration: order must be 'blocked', 'interleaved' or 'random', got {order!r}")

        trial_info = []

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / 'verb_generation' / 'verb_generation.csv')

        stim = stim.sample(frac=1).reset_index(drop=True)

        t = 0

        for n in range(n_trials):
            selected_stim = stim.iloc[n]['word']
            trial = {}
            trial['trial_num'] = n
            trial['condition'] = trial_conditions[n]
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial['stim'] = selected_stim
            trial['display_trial_feedback'] = False
            trial_info.append(trial)

            # Update for next trial:
            t= trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name,sep='\t',index=False)

        return trial_info

class TongueMovement(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'tongue_movement'

    def make_task_file(self,
                        task_dur =  30,
                        trial_dur = 1,
                        iti_dur   = 0,
                        file_name = None):
        """
        Create a tongue-movement task file.

        Args:
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each tongue-movement cycle in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            # Alternate between 'right' and 'left' for each trial
            trial['trial_type'] = 'right' if n % 2 == 0 else 'left'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return trial_info

class AuditoryNarrative(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'auditory_narrative'

    def make_task_file(self,
                       task_dur=30,
                       trial_dur=30,
                       iti_dur=0,
                       file_name=None,
                       run_number=None):
        """
        Create an auditory-narrative task file. Each run plays a distinct
        narrative clip (narrative_NN.wav) selected by run_number.

        Args:
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each audio clip in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.
            run_number (int): Run number, used to select which audio clip to play.

        Returns:
            pd.DataFrame: Task information as a DataFrame.

        Raises:
            ValueError: If run_number is None or exceeds the number of available
                narrative clips.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        # Each run uses a distinct narrative (narrative_NN.wav) - novelty matters error if no enough files
        available = sorted((self.stim_dir / self.name).glob('narrative_[0-9][0-9].wav'))
        if run_number is None or run_number > len(available):
            raise ValueError(
                f"AuditoryNarrative: only {len(available)} narratives available; "
                f"cannot generate run {run_number}. Add more narrative_NN.wav files "
                f"or reduce the number of runs.")

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            trial['stim'] = f'narrative_{run_number:02d}.wav'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class SpatialNavigation(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'spatial_navigation'
        self.location_pairs = [('FRONT-DOOR', 'LIVING-ROOM'), ('WASHROOM', 'LIVING-ROOM'), ('BEDROOM', 'LIVING-ROOM'),
                        ('KITCHEN', 'LIVING-ROOM'), ('FRONT-DOOR', 'WASHROOM'), ('BEDROOM', 'FRONT-DOOR'),
                        ('KITCHEN', 'FRONT-DOOR'), ('KITCHEN', 'BEDROOM'), ('KITCHEN', 'WASHROOM'),
                        ('BEDROOM', 'WASHROOM')]


    def make_task_file(self,
                       task_dur=30,
                       trial_dur=30,
                       iti_dur=0,
                       file_name=None,
                       run_number=None):
        """
        Create a spatial-navigation task file (imagined navigation between two
        remembered locations).

        Args:
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of the imagination period in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.
            run_number (int): Run number, used to select which location pair to use.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            #loc1, loc2 = self.location_pairs[run_number - 1]
            loc1, loc2 = self.location_pairs[(run_number - 1) % len(self.location_pairs)]
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial['location_1'] = loc1
            trial['location_2'] = loc2
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class TheoryOfMind(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'theory_of_mind'
        self.matching_stimuli = False # stimuli for active condition (belief) are different from stimuli for passive condition (photo)

    def make_task_file(self, hand='right',
                       responses = [1,2], # 1 = True, 2 = False
                       run_number=None,
                        task_dur=30,
                        trial_dur=14,
                        iti_dur=1,
                        story_dur=10,
                        question_dur=4,
                        text_height=1.25,
                        file_name=None,
                        stim_file=None,
                        condition=None):
        """
        Create a theory-of-mind task file (story followed by a true/false statement).

        Args:
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys for [True, False].
            run_number (int): Run number, used to select stimuli for that run.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Total duration of each trial in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            story_dur (float): Duration the story is displayed in seconds.
            question_dur (float): Duration the question is displayed in seconds.
            text_height (float): Height of the story/question text in degrees of visual angle.
            file_name (str): Name of the file to save the task data.
            stim_file (str): Optional path to a custom stimulus CSV.
            condition (str): If set, only trials of this condition are included
                ('belief' or 'photo').

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        # Count number of trials based on timing; may be overridden below when an
        # explicit stimulus list is provided (so distribution, not timing, sets
        # the exact trial count).
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

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

class PassageListening(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'passage_listening'

    def make_task_file(self,
                       run_number,
                       condition='intact',
                       task_dur=30,
                       trial_dur=14.5,
                       iti_dur=0.5,
                       file_name=None,
                       stim_file=None,):
        """
        Create a passage-listening task file (intact vs degraded speech).

        Args:
            run_number (int): Run number, used to select which passages to play.
            condition (str): Which condition to include ('intact' or 'degraded').
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each passage in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.
            stim_file (str): Optional path to a custom stimulus CSV.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))

        # Load the audio/condition table and keep only the requested condition
        stim = pd.read_csv(self.stim_dir / self.name / (stim_file or f'{self.name}.csv'))
        valid = sorted (stim['condition'].unique()) # check if the condition is there
        if condition not in valid:
            raise ValueError(f"PassageListening: unknown condition {condition!r} (expected one of {valid})")
        stim = stim[stim['condition'] == condition].reset_index(drop=True)
        # Wrap around the available passages if more runs are requested than there are.
        total = len(stim)
        if (run_number - 1) * n_trials >= total:
            print(f"Warning: PassageListening only has {total} '{condition}' passages; "
                  f"run {run_number} wraps around and repeats stimuli.")
        idx = [((run_number - 1) * n_trials + k) % total for k in range(n_trials)]
        stim = stim.iloc[idx].reset_index(drop=True)

        trial_info = []
        t = 0
        for n in range(len(stim)):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['condition'] = stim['condition'][n]
            trial['stim'] = stim['audio'][n]
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class ActionObservation(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'action_observation'
        self.matching_stimuli = False # stimuli for active condition (knot tying) are different from stimuli for passive condition (knot watching)

        # Medium/bad knot vids
        # self.knot_names = [
        #                 'Ampere', 'Arbor', 'Baron', 'Belfry', 'Bramble', 'Chamois', 'Coffer',
        #                 'Farthing', 'Fissure', 'Gentry', 'Henchman', 'Magnate', 'Perry', 'Phial', 'Polka',
        #                 'Rosin', 'Shilling', 'Simper', 'Spangle', 'Squire', 'Vestment', 'Wampum', 'Wicket'
        #             ]

        # good knot vids
        self.knot_names = ['Adage',
                        'Brigand', 'Brocade', 'Casement',  'Cornice',\
                        'Flora', 'Frontage', 'Gadfly', 'Garret', \
                        'Mutton','Placard', 'Purser']

    def make_task_file(self,
                        run_number = None,
                        task_dur=30,
                        trial_dur=14,
                        iti_dur=1,
                        file_name=None):
        """
        Create an action-observation task file (knot-tying videos).

        Args:
            run_number (int): Run number, used to select which knot stimulus to show.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each video trial in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            knot_index = (run_number - 1) % len(self.knot_names)
            # 'condition' is a real per-trial column, so a hand-written task file
            # can order the conditions however it likes. The make_task_file default
            # is the action clip first, then control.
            if n == 0:
                trial['condition'] = 'action'
                trial['stim'] = f'knotAction{self.knot_names[knot_index]}.mov'
            else:
                trial['condition'] = 'control'
                trial['stim'] = f'knotControl{self.knot_names[knot_index]}.mov'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class DemandGrid(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'demand_grid'

    def get_adjacent_positions(self,pos, grid_size):
        """
        Get all adjacent positions within the grid boundaries.

        Args:
            pos (tuple): Current position (x, y).
            grid_size (tuple): Size of the grid (rows, cols).

        Returns:
            list: List of adjacent positions.
        """
        x, y = pos
        adjacent = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= x + dx < grid_size[0] and 0 <= y + dy < grid_size[1]
        ]
        return adjacent

    def generate_sequence(self, grid_size, num_steps, num_boxes_lit):
        """
        Generate the original sequence of lit-up boxes, ensuring adjacency between boxes
        in each step and no reuse of positions across the entire sequence.

        Args:
            grid_size (tuple): Size of the grid (rows, cols).
            num_steps (int): Number of steps in the sequence.
            num_boxes_lit (int): Number of boxes lit up per step.

        Returns:
            list: Original sequence of steps, where each step is a list of positions.
        """
        sequence = []
        used_positions = set()  # Tracks all positions that have been used

        for _ in range(num_steps):
            step = []
            if not sequence:
                # First step: Randomly choose starting positions
                while len(step) < num_boxes_lit:
                    pos = (
                        random.randint(0, grid_size[0] - 1),
                        random.randint(0, grid_size[1] - 1)
                    )
                    if pos not in used_positions:
                        step.append(pos)
                        used_positions.add(pos)
            else:
                # Subsequent steps: Ensure adjacency and uniqueness
                available_positions = {
                    adj
                    for prev_step in sequence
                    for pos in prev_step
                    for adj in self.get_adjacent_positions(pos, grid_size)
                }
                available_positions -= used_positions  # Remove already-used positions

                while len(step) < num_boxes_lit:
                    if not available_positions:
                        raise ValueError("not enough valid adjacent positions available")
                    pos = random.choice(list(available_positions))
                    step.append(pos)
                    used_positions.add(pos)
                    available_positions.remove(pos)  # Avoid duplicates in the current step

            sequence.append(step)

        return sequence

    @staticmethod
    def _count_connected_components(positions):
        """Count connected components in a set of grid positions (4-directional)."""
        from collections import deque
        if not positions:
            return 0
        remaining = set(positions)
        components = 0
        while remaining:
            components += 1
            start = next(iter(remaining))
            queue = deque([start])
            remaining.remove(start)
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (x + dx, y + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
        return components

    def modify_sequence(self, sequence, grid_size):
        """
        Modify the original sequence to create a new sequence for comparison,
        ensuring adjacency and uniqueness within the modified step. If a step
        cannot be modified due to lack of valid adjacent positions, try another step.

        The modified sequence is constrained to have the same number of connected
        components as the original, so the distractor is not visually distinguishable
        by spatial continuity alone.

        Args:
            sequence (list): Original sequence of steps.
            grid_size (tuple): Size of the grid (rows, cols).

        Returns:
            list: Modified sequence of steps.
        """
        original_positions = [pos for step in sequence for pos in step]
        original_cc = self._count_connected_components(original_positions)

        modified_sequence = sequence[:]
        available_step_indices = list(range(len(sequence)))  # List of all step indices to try

        while available_step_indices:
            random_step_index = random.choice(available_step_indices)  # Choose a random step
            original_step = sequence[random_step_index]

            # Gather all used positions from the entire sequence
            used_positions = {t for step in sequence for t in step}

            # Use all remaining steps (not the step being replaced) as the adjacency
            # source, matching the rule used by generate_sequence().
            remaining_steps = [s for i, s in enumerate(sequence) if i != random_step_index]

            new_step = []
            try:
                while len(new_step) < len(original_step):
                    available_positions = {
                        adj
                        for step in remaining_steps
                        for pos in step
                        for adj in self.get_adjacent_positions(pos, grid_size)
                    }
                    available_positions -= used_positions  # Exclude already-used positions
                    available_positions -= set(new_step)  # Exclude positions already added to the new step

                    if not available_positions:
                        raise ValueError("Not enough valid adjacent positions available for modification.")

                    # Choose a random position from the available ones
                    new_pos = random.choice(list(available_positions))
                    new_step.append(new_pos)
                    used_positions.add(new_pos)  # Mark the position as used

                # Check that the modified grid has the same connectivity as the original.
                # Reject if the replacement creates extra disconnected components.
                candidate = sequence[:]
                candidate[random_step_index] = new_step
                candidate_positions = [pos for step in candidate for pos in step]
                candidate_cc = self._count_connected_components(candidate_positions)

                if candidate_cc <= original_cc:
                    modified_sequence[random_step_index] = new_step
                    return modified_sequence
                else:
                    raise ValueError("Modified sequence has more connected components than original.")

            except ValueError:
                # Remove this step index from the list of available steps to try
                available_step_indices.remove(random_step_index)
                continue  # Try another step

        # If all steps fail, raise an error
        raise ValueError("No valid step could be modified with the given constraints.")


    def make_task_file(self,
                   hand='right',
                   responses=[1, 2],  # 1 = Left, 2 = Right
                   grid_size=(3, 4),
                   num_steps=3,
                   num_boxes_lit=2,
                   task_dur=30,
                   trial_dur=7,
                   question_dur=3,
                   sequence_dur=4,
                   iti_dur=0.5,
                   file_name=None):
        """
        Create a task file with the specified parameters.

        Args:
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys for left and right.
            grid_size (tuple): Size of the grid (rows, cols).
            num_steps (int): Number of steps in the sequence.
            num_boxes_lit (int): Number of boxes lit up per step.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each trial.
            question_dur (float): Duration of the question phase.
            sequence_dur (float): Duration of the sequence presentation phase.
            iti_dur (float): Inter-trial interval duration.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(task_dur / (trial_dur + iti_dur))
        trial_info = []
        current_time = 0

        for n in range(n_trials):
            while True:  # Retry logic
                try:
                    # Generate the original sequence
                    original_sequence = self.generate_sequence(grid_size, num_steps, num_boxes_lit)
                    # Attempt to create a modified sequence
                    modified_sequence = self.modify_sequence(original_sequence, grid_size=grid_size)
                    break
                except ValueError:
                    continue

            correct_side = random.choice(['left', 'right'])
            trial_type = 0 if correct_side == 'left' else 1
            trial = {
                'key_left': responses[0],
                'key_right': responses[1],
                'correct_side': correct_side,
                'trial_type': trial_type,
                'trial_num': n,
                'hand': hand,
                'grid_size': grid_size,
                'num_steps': num_steps,
                'original_sequence': list(itertools.chain.from_iterable(original_sequence)),
                'modified_sequence': list(itertools.chain.from_iterable(modified_sequence))  ,
                **{f'original_step_{i+1}': step for i, step in enumerate(original_sequence)},  # Unpack original steps (each grid shown in the exp)
                'display_trial_feedback': True,
                'trial_dur': trial_dur,
                'sequence_dur': sequence_dur,
                'question_dur': question_dur,
                'iti_dur': iti_dur,
                'start_time': current_time,
                'end_time': current_time + trial_dur + iti_dur
            }
            trial_info.append(trial)

            # Update for the next trial:
            current_time = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class Reading(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'reading'

    def make_task_file(self,
                        run_number = None,
                        condition = 'sentences',   # 'sentence' or 'nonwords'
                        task_dur=30,
                        trial_dur=5.8,
                        iti_dur=0.2,
                        file_name=None,
                        stim_file=None):
        """
        Create a reading task file (sentences or nonwords, shown word by word).

        Args:
            run_number (int): Run number, used to select which sentences to show.
            condition (str): 'sentences' or 'nonwords'.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each sentence presentation in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.
            stim_file (str): Optional path to a custom stimulus CSV.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        # Select the stimulus list for the requested condition (sentences vs nonwords).
        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            if condition == 'sentences':
                csv = 'sentences_shuffled.csv'

            elif condition == 'nonwords':
                csv = 'nonwords_shuffled.csv'
            else:
                raise ValueError( F" task Reading: unknown condition {condition!r} (expected 'sentences' or 'nonwords')")
            stim = pd.read_csv(self.stim_dir / self.name / csv)

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['condition'] = condition
            sentence_index = (run_number - 1) * n_trials + n
            trial['stim'] = stim['sentence'][sentence_index]
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class OddBall(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'oddball'


    def make_task_file(self,
                    hand = 'right',
                    responses = [1,2], # 1,2 any press is ok
                    task_dur=30,
                    trial_dur=0.15,
                    iti_dur=0.85,
                    file_name=None):
        """
        Create an oddball-detection task file (respond only to a red 'K').

        Args:
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration the stimulus is displayed in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        stimuli = ['black_K'] * 12 + ['black_O'] * 12 + ['red_K'] * 3 + ['red_O'] * 3

        # shuffle stimuli
        random.shuffle(stimuli)

        t = 0

        for n in range(len(stimuli)):
            trial = {}
            trial['key_one'] = responses[0]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['trial_type'] = 1 if stimuli[n] == 'red_K' else 0
            trial['stim'] = stimuli[n]
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class FingerSequence(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'finger_sequence'
        self.matching_stimuli = False # sequence of numbers are different for easy and hard sequence condition

    def generate_sequence(self):
        sequence = [random.choice([1, 2, 3, 4])]
        while len(sequence) < 6:
            next_digit = random.choice([d for d in [1, 2, 3, 4] if d != sequence[-1]])
            sequence.append(next_digit)
        return ' '.join(map(str, sequence))

    def make_task_file(self,
                        hand = 'bimanual',
                        task_dur=30,
                        trial_dur=3.25,
                        iti_dur=0.5,
                        file_name=None):
        """
        Create a finger-sequence task file (press a 6-digit sequence in order).
        Each digit (1-4) is the finger/key to press; scoring compares the pressed
        key number directly to the sequence digit.

        Args:
            hand (str): Hand(s) used for response ('bimanual', 'right', or 'left').
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each trial in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = True
            # choose random sequence
            trial['stim'] = self.generate_sequence()
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class FlexionExtension(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'flexion_extension'

    def make_task_file(self,
                        task_dur = 30,
                        stim_dur = 2,
                        file_name = None):
        """
        Create a flexion-extension (toe movement) task file. The block is paced
        by a cue that alternates between 'flexion' and 'extension' every stim_dur
        seconds, written as one row per cue. The runtime just shows each cue for
        its duration.

        Args:
            task_dur (float): Total duration of the block in seconds.
            stim_dur (float): Duration each cue ('flexion'/'extension') is shown.
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        cues = ['flexion', 'extension']
        n_phases = int(np.floor(task_dur / stim_dur))
        trial_info = []
        t = 0
        for n in range(n_phases):
            trial_info.append({
                'trial_num': n,
                'stim': cues[n % 2],
                'trial_dur': stim_dur,
                'start_time': t,
                'end_time': t + stim_dur,
            })
            t += stim_dur

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return trial_info

class SemanticPrediction(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'semantic_prediction'

    def make_task_file(self, hand='right',
                       responses = [1,2], # 1 = True, 2 = False
                       run_number=None,
                       task_dur=30,
                        trial_dur=15,
                        sentence_dur=2,
                        file_name=None,
                        stim_file=None,
                        stim=None):
        """
        Create a semantic-prediction task file (judge whether the final word
        makes the sentence meaningful).

        Args:
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys for [meaningful, meaningless].
            run_number (int): Run number, used to select which sentences to present.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Total duration budgeted for each trial in seconds.
            sentence_dur (float): Response window for the final word, in seconds.
            file_name (str): Name of the file to save the task data.
            stim_file (str): Optional path to a custom stimulus CSV.
            stim (pd.DataFrame): Optional pre-loaded stimulus table.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur)))
        trial_info = []
        t = 0

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / 'semantic_prediction' / 'semantic_prediction.csv')

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
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['sentence_dur'] = sentence_dur
            trial['sentence'] = stim['sentence'].iloc[n]
            trial['trial_type'] = random.choice([0,1]) # 0 = meaningless, 1 = meaningful; randomize trial type for each sentence
            last_word = [stim['wrong_word'].iloc[n], stim['right_word'].iloc[n]]
            trial['last_word'] = last_word[trial['trial_type']]
            trial['display_trial_feedback'] = True
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur

            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return trial_info

class VisualSearch(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'visual_search'

    def make_task_file(self,
                        hand = 'right',  #to recode for alternating hands: put left here, and put 3,4 in responses
                        responses = [1,2], # 1 = match, 2 = no match
                        task_dur =  30,
                        trial_dur = 2,
                        iti_dur   = 0.5,
                        easy_prob=0.5,
                        file_name = None ):
        """
        Create a visual-search task file (find a canonically-oriented 'L').

        Args:
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys for [target present, target absent].
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each trial in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            easy_prob (float): Probability of an easy trial (4 stimuli vs. 8).
            file_name (str): Name of the file to save the task data.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []
        t = 0

        for n in range(n_trials):
            trial = {}
            trial['key_true'] = responses[0]
            trial['key_false'] = responses[1]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = True
            trial['trial_type'] = random.choice([0,1])
            # Difficulty is the condition: the number of search items (set size).
            trial['condition'] = '4-items' if random.random() < easy_prob else '8-items'
            trial['feedback_type'] = 'acc'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name,sep='\t',index=False)
        return trial_info


class RMET(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'rmet'
        self.matching_stimuli = True # stimuli are matched for the active condition (determine emotion) and passive condition (determine age)
        self.half_assigned = True
        self.repeat_stimuli_from_previous_runs = True

    def make_task_file(self, hand='right',
                        responses = [1,2,3,4],
                        run_number=None,
                        task_dur=30,
                        trial_dur=6,
                        iti_dur=1.5,
                        option_text_height=1.2,
                        option_position_scale=1.0,
                        picture_scale=0.7,
                        show_last_seconds=0,
                        file_name=None,
                        stim_file = None,
                        condition=None,
                        half=None,
                        stim=None):
        """
        Create an RMET task file (Reading the Mind in the Eyes; emotion or age).

        Args:
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys mapped to the four options.
            run_number (int): Run number, used to select which stimuli to present.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration each stimulus is displayed in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            option_text_height (float): Height of the answer-option text in degrees of visual angle.
            option_position_scale (float): Spatial scaling for option positions (<1 brings them closer).
            picture_scale (float): Scaling of the eye-region image (>1 enlarges).
            show_last_seconds (float): If >0, show options only for the final N seconds of the trial.
            file_name (str): Name of the file to save the task data.
            stim_file (str): Optional path to a custom stimulus CSV.
            condition (str): If set, only trials of this condition are included
                ('emotion' or 'age').
            half (str): Optional split of the stimulus set into halves.
            stim (pd.DataFrame): Optional pre-loaded stimulus table.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        if stim_file:
            stim = pd.read_csv(self.stim_dir / self.name / stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}.csv')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[
                ~stim['condition'].str.contains('practice', na=False)
                & (stim['condition'].astype(str).str.lower() != 'exclude')
            ]

        # Ignore stim_list and stimulus_seed: selection is driven entirely by
        # the provided stim_file (if any), half-based slicing, or run_number.
        if half: # Selects different stimuli for the social and control condition, to enable showing each story only once for each participant
            start_row = (run_number - 1) * n_trials
            end_row = run_number * n_trials - 1
            stim_half = stim[stim['half'] == half]
            if n_trials <= stim_half.iloc[start_row:end_row + 1].shape[0]: # Check if there are enough stimuli for the run
                stim = stim_half.iloc[start_row:end_row + 1].reset_index(drop=True)
            elif self.repeat_stimuli_from_previous_runs:
                # If there are not enough stimuli, repeat stimuli from the OTHER condition of previous runs
                new_run_number = run_number - 4
                new_start_row = (new_run_number - 1) * n_trials
                new_end_row = new_run_number * n_trials - 1
                stim = stim[stim['half'] != half].iloc[new_start_row:new_end_row + 1].reset_index(drop=True)
            else:
                raise ValueError('Not enough stimuli for the run')
        else:
            start_row = (run_number - 1) * n_trials
            end_row = run_number * n_trials - 1
            stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        n_actual = min(n_trials, len(stim))
        for n in range(n_actual):
            trial = {}
            trial['key_one'] = responses[0]
            trial['key_two'] = responses[1]
            trial['key_three'] = responses[2]
            trial['key_four'] = responses[3]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['option_text_height'] = option_text_height
            trial['option_position_scale'] = option_position_scale
            trial['picture_scale'] = picture_scale
            trial['stim'] = stim['picture'][n]
            trial['options'] = stim['options'][n]
            trial['condition'] = stim['condition'][n]
            trial['answer'] = stim['answer'][n]
            trial['display_trial_feedback'] = True
            if show_last_seconds:
                trial['show_last_seconds'] = show_last_seconds
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return trial_info


class Movie(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'movie'
        self.matching_stimuli = False # Romance movie clips are different from nature movie clips and landscape movie clips

    def make_task_file(self,
                       run_number = None ,
                       task_dur=30,
                       trial_dur=30,
                       iti_dur=0,
                       file_name=None,
                       stim_file=None,
                       condition=None,
                       media_scale=0.4):
        """
        Create a movie-watching task file (passive viewing of a 30s clip).

        Args:
            run_number (int): Run number, used to select which clip to show.
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration of each clip in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.
            stim_file (str): Optional path to a custom stimulus CSV.
            condition (str): Which clips to use ('romance', 'nature', or
                'landscape'). If None, all non-practice clips are used.
            media_scale (float): Clip width as a fraction of the window width.

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        if stim_file:
            stim = pd.read_csv(self.stim_dir / self.name / stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}.csv')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[
                ~stim['condition'].str.contains('practice', na=False)
                & (stim['condition'].astype(str).str.lower() != 'exclude')
            ]

        # Wrap around the available clips if more runs are requested than there are
        # stimuli (e.g. landscape has only 10 clips).
        total = len(stim)
        if (run_number - 1) * n_trials >= total:
            cond_label = condition if condition else 'selected'
            print(f"Warning: Movie only has {total} '{cond_label}' clips; "
                  f"run {run_number} wraps around and repeats clips.")
        idx = [((run_number - 1) * n_trials + k) % total for k in range(n_trials)]
        stim = stim.iloc[idx].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            trial['stim'] = stim['video'][n]
            trial['condition'] = stim['condition'][n]
            trial['media_scale'] = media_scale
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class FauxPas(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'faux_pas'
        self.matching_stimuli = True
        self.half_assigned = True
        self.repeat_stimuli_from_previous_runs = True

    def make_task_file(self, hand='right',
                    responses = [1,2], # 1 = True, 2 = False
                    run_number=None,
                    task_dur=30,
                    trial_dur=14,
                    iti_dur=1,
                    story_dur=10,
                    question1_dur=4,
                    text_height=1.25,
                    file_name=None,
                    stim_file=None,
                    condition=None,
                    half=None,
                    stimulus_seed=None,
                    exclude_stimuli=None):


        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}.csv')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[
                ~stim['condition'].str.contains('practice', na=False)
                & (stim['condition'].astype(str).str.lower() != 'exclude')
            ]
            if stimulus_seed is not None:
                if exclude_stimuli is not None:
                    stim = stim[~stim['story'].isin(exclude_stimuli)]
                stim = stim.sample(n=min(n_trials, len(stim)), random_state=stimulus_seed).reset_index(drop=True)

        if stimulus_seed is not None:
            if condition is not None:
                if exclude_stimuli is not None:
                    stim = stim[~stim['story'].isin(exclude_stimuli)]
                stim = stim.sample(n=n_trials, random_state=stimulus_seed).reset_index(drop=True)
        elif half: # Selects different stimuli for the social and control condition, to enable showing each story only once for each participant
            start_row = (run_number - 1) * n_trials
            end_row = run_number * n_trials - 1
            stim_half = stim[stim['half'] == half]
            if n_trials <= stim_half.iloc[start_row:end_row + 1].shape[0]: # Check if there are enough stimuli for the run
                stim = stim_half.iloc[start_row:end_row + 1].reset_index(drop=True)
            elif self.repeat_stimuli_from_previous_runs:
                # If there are not enough stimuli, repeat stimuli from the OTHER condition of previous runs
                new_run_number = run_number - 5
                new_start_row = (new_run_number - 1) * n_trials
                new_end_row = new_run_number * n_trials - 1
                stim = stim[stim['half'] != half].iloc[new_start_row:new_end_row + 1].reset_index(drop=True)
            else:
                raise ValueError('Not enough stimuli for the run')
        else:
            start_row = (run_number - 1) * n_trials
            end_row = run_number * n_trials - 1
            stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['key_yes'] = responses[0]
            trial['key_no'] = responses[1]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['story'] = stim['story'][n]
            trial['question'] = stim['question1'][n]
            trial['options'] = stim['options1'][n]
            if str(stim['answer1'][n]) == 'Yes':
                trial['trial_type'] = 1
            else:
                trial['trial_type'] = 2
            trial['condition'] = stim['condition'][n]
            trial['story_dur'] = story_dur
            trial['question_dur'] = question1_dur
            trial['text_height'] = text_height
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

class Affective(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'affective'

    def make_task_file(self,
                       task_dur=30,
                       trial_dur=1.6,
                       iti_dur=0.4,
                       file_name=None,
                       hand='right',
                       responses=[1,2]):
        """
        Create an affective-picture task file (judge pleasant vs unpleasant).

        Args:
            task_dur (float): Total task duration in seconds.
            trial_dur (float): Duration each image is displayed in seconds.
            iti_dur (float): Inter-trial interval duration in seconds.
            file_name (str): Name of the file to save the task data.
            hand (str): Hand used for response ('right' or 'left').
            responses (list): Response keys for [unpleasant, pleasant].

        Returns:
            pd.DataFrame: Task information as a DataFrame.
        """
        # check how many trials to include
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        n_pleasant = n_trials // 2
        n_unpleasant = n_trials - n_pleasant

        # Randomly sample numbers 1–26 (image name numbers)
        pleasant_nums = random.sample(range(1, 27), n_pleasant)
        unpleasant_nums = random.sample(range(1, 27), n_unpleasant)

        # Create stim list
        stim = [{'imgName': f'pleasant{n}.jpg', 'trialType': 2} for n in pleasant_nums] + \
               [{'imgName': f'unpleasant{n}.jpg', 'trialType': 1} for n in unpleasant_nums]

        random.shuffle(stim)  # mix trial order

        # Build trial list
        trial_info = []
        t = 0
        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['stim'] = stim[n]['imgName']
            trial['trial_type'] = stim[n]['trialType']
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['key_unpleasant'] = responses[0]
            trial['key_pleasant'] = responses[1]
            trial['display_trial_feedback'] = True
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)

        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class SerialReactionTime(TaskFile):
      def __init__(self, const):
          super().__init__(const)
          self.name = 'serial_reaction_time'

      def make_task_file(self,
                         hand='bimanual',
                         task_dur=30,
                         initial_wait=1.0,
                         trial_dur=0.5,
                         iti_dur=1.0,
                         file_name=None):
          # wait one second at the start
          effective_dur = task_dur - initial_wait
          n_trials = int(np.floor(effective_dur / (trial_dur + iti_dur)))
          trial_info = []

          t = initial_wait
          prev_stim = 0

          for n in range(n_trials):
              trial = {}
              trial['trial_num'] = n
              trial['hand'] = hand
              trial['trial_dur'] = trial_dur
              trial['iti_dur'] = iti_dur

            # Ensure the same stimulus doesn't appear on consecutive trials
              stim = prev_stim
              while stim == prev_stim:
                  stim = random.randint(1, 4)
              trial['stim'] = stim
              prev_stim = stim

              trial['start_time'] = t
              trial['end_time'] = t + trial_dur + iti_dur
              trial_info.append(trial)
              t = trial['end_time']

          trial_info = pd.DataFrame(trial_info)
          if file_name is not None:
              ut.dircheck(self.task_dir / self.name)
              trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

          return trial_info


class FingerRhythmic(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'finger_rhythmic'

    def make_task_file(self,
                       hand='right',
                       responses=[1],
                       run_number= None,
                       task_dur = 70,
                       trial_dur=35, # 2 sec trial start text, 27.95 sec tone train, ~5 sec buffer
                       iti_dur=0,
                       ioi=0.6,  # Inter-onset interval in seconds (default: 600ms). Can be single value or list for different pace levels
                       file_name=None):

        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        # Handle ioi as single value or list
        if isinstance(ioi, (list, np.ndarray)):
            ioi_list = list(ioi)
            # Expand to cover all trials by repeating the list
            ioi_list_expanded = (ioi_list * ((n_trials // len(ioi_list)) + 1))[:n_trials]
            # Randomize the order
            random.shuffle(ioi_list_expanded)
        else:
            ioi_list_expanded = [ioi] * n_trials  # Repeat single value for all trials

        for n in range(n_trials):
            trial = {}
            trial['key_one'] = responses[0]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['ioi'] = ioi_list_expanded[n]  # Use randomized IOI assignment
            trial['stim'] = 'generated'
            trial['display_trial_feedback'] = False
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class TimePerception(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'time_perception'  # folder: stimuli/perception/, tasks/perception/

    def make_task_file(self,
                       modality='time',          # 'time' or 'volume'
                       responses=[1, 2],         # code 1 = left option, 2 = right option
                       n_trials= 30,              # must be even
                       trial_dur=4,            # tone + question window duration
                       iti_dur=1.0,
                       question_dur=2.0,
                       display_feedback= True,
                       run_number=None,
                       file_name=None,
                       **unused):

        # sides per modality
        if modality == 'time':
            left_label, right_label = 'shorter', 'longer'
        elif modality == 'volume':
            left_label, right_label = 'quieter', 'louder'

        sides = [left_label] * (n_trials // 2) + [right_label] * (n_trials // 2)
        np.random.default_rng(run_number).shuffle(sides)

        rows, t = [], 0.0
        for i, side in enumerate(sides):
            rows.append(dict(
                trial_num=i,
                modality=modality,                   # drives Task branching
                side=side,                           # shorter/longer or softer/louder
                key_one=int(responses[0]),           # instruction mapping only
                key_two=int(responses[1]),
                trial_type=1 if side in (left_label,) else 2,  # correct code
                question_dur=float(question_dur),
                trial_dur=float(trial_dur),
                iti_dur=float(iti_dur),
                display_trial_feedback= display_feedback,

                # runtime logs (filled in Task)
                comparison_ms=np.nan,
                comparison_dba=np.nan,

                start_time=float(t),
                end_time=float(t + trial_dur + iti_dur),
            ))
            t = rows[-1]['end_time']

        df = pd.DataFrame(rows)
        if file_name:
            ut.dircheck(self.task_dir / self.name)
            df.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return df

class SensMotControl(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'sensmot_control'

    def make_task_file(self,
                       hand='right',
                       responses=[1, 2],
                       run_number=None,
                       task_dur=300,
                       trial_dur=3,
                       question_dur=2,
                       iti_dur= 1,
                       file_name=None,
                       stim_file = None,
                       condition=None):

        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        if stim_file:
            stim = pd.read_csv(self.stim_dir / self.name / stim_file, sep='\t')
        else:
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}_block1.csv', sep='\t')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[
                ~stim['condition'].str.contains('practice', na=False)
                & (stim['condition'].astype(str).str.lower() != 'exclude')
            ]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['key_one'] = responses[0]
            trial['key_two'] = responses[1]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['question_dur'] = question_dur
            trial['iti_dur'] = iti_dur
            trial['trial_type'] = stim['corr_resp'][n]
            trial['stim'] = stim['color'][n]
            trial['condition'] = stim['condition'][n]
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




