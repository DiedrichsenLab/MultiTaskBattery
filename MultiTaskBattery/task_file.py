#%%
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
                  keep_in_middle=None):
    """
    Make a single run file
    """
    # Get rows of the task_table corresponding to the task_list
    indx = [np.where(ut.task_table['name']==t)[0][0] for t in task_list]
    R = {'task_name':task_list,
         'task_code':ut.task_table['code'].iloc[indx],
         'task_file':tfiles,
         'instruction_dur':[instruction_dur]*len(task_list)}
    R = pd.DataFrame(R)
    R = shuffle_rows(R, keep_in_middle=keep_in_middle)
    R = add_start_end_times(R, offset, task_dur+instruction_dur, run_time=run_time)
    return R

def get_task_class(name):
    """Creates an object of the task class based on the task name
    Args:
        name (str): name of the task
    Returns:
        class_name (str): class name for task
    """
    index = np.where(ut.task_table['name']==name)[0][0]
    class_name = ut.task_table.iloc[index]['task_class']
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


class NBack(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'n_back'

    def make_task_file(self,
                       hand = 'right',
                       responses = [1,2], # 1 = match, 2 = no match
                       task_dur = 30,
                       trial_dur = 2,
                       iti_dur   = 0.5,
                       stim = ['9.jpg','11.jpg','18.jpg','28.jpg'],
                       file_name = None ):
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        prev_stim = ['x','x']
        t = 0
        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = True
            trial['key_match'] = responses[0]
            trial['key_nomatch'] = responses[1]
            # Determine if this should be N-2 repetition trial

            if n<2:
                trial['trial_type'] = 0
            else:
                trial['trial_type'] = np.random.randint(0,2)
            # Now choose the stimulus accordingly: avoid any reps
            if trial['trial_type']==0:
                trial['stim'] = prev_stim[1]
                while (trial['stim'] == prev_stim[0]) | (trial['stim'] == prev_stim[1]):
                    trial['stim'] = stim[np.random.randint(0,len(stim))]
            else:
                trial['stim'] = prev_stim[1]

            trial['display_trial_feedback'] = True
            trial['feedback_type'] = 'acc'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t= trial['end_time']
            prev_stim[1] = prev_stim[0]
            prev_stim[0] = trial['stim']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name,sep='\t',index=False)
        return trial_info

class Rest(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'rest'

    def make_task_file(self,
                       task_dur = 30,
                       file_name = None):
        trial = {}
        trial['trial_num'] = [1]
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
                       task_dur = 30,
                       trial_dur = 2,
                       iti_dur   = 0.5,
                       file_name = None,
                       stim_file = None):
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
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

            # Determine if this is a read or generate trial
            if n < n_trials/2:
                trial['trial_type'] = 'read'
            else:
                trial['trial_type'] = 'generate'
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
                       task_dur = 30,
                       trial_dur = 1,
                       iti_dur   = 0,
                       file_name = None):
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
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
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

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

        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            loc1, loc2 = self.location_pairs[run_number - 1]
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
                       task_dur=300,
                       trial_dur=14,
                       iti_dur=1,
                       story_dur=10,
                       question_dur=4,
                       file_name=None,
                       stim_file=None,
                       condition=None):

        # count number of trials
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
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
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

class DegradedPassage(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'degraded_passage'

    def make_task_file(self,
                       run_number = None,
                       task_dur=30,
                       trial_dur=14.5,
                       iti_dur=0.5,
                       file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            # Select the appropriate audio file
            audio_file_num = (run_number - 1) * n_trials + n + 1
            trial['stim'] = f'degraded_passage_{audio_file_num}.wav'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class IntactPassage(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'intact_passage'

    def make_task_file(self,
                       run_number,
                       task_dur=30,
                       trial_dur=14.5,
                       iti_dur=0.5,
                       file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            # Select the appropriate audio file
            audio_file_num = (run_number - 1) * n_trials + n + 1
            trial['stim'] = f'intact_passage_{audio_file_num}.wav'
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
        # self.knot_names = ['Adage',
        #                 'Brigand', 'Brocade', 'Casement',  'Cornice',\
        #                 'Flora', 'Frontage', 'Gadfly', 'Garret', \
        #                 'Mutton','Placard', 'Purser']

    def make_task_file(self,
                       run_number = None,
                       task_dur=30,
                       trial_dur=14,
                       iti_dur=1,
                       file_name=None,
                       knot_names = ['Adage',
                                     'Brigand', 'Brocade', 'Casement',  'Cornice', \
                                     'Flora', 'Frontage', 'Gadfly', 'Garret', \
                                     'Mutton','Placard', 'Purser']):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = None
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            knot_index = (run_number - 1)
            if n == 0:
                trial['stim'] = f'knotAction{knot_names[knot_index]}.mov'
            else:
                trial['stim'] = f'knotControl{knot_names[knot_index]}.mov'
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

    def modify_sequence(self, sequence, grid_size):
        """
        Modify the original sequence to create a new sequence for comparison,
        ensuring adjacency and uniqueness within the modified step. If a step
        cannot be modified due to lack of valid adjacent positions, try another step.

        Args:
            sequence (list): Original sequence of steps.
            grid_size (tuple): Size of the grid (rows, cols).

        Returns:
            list: Modified sequence of steps.
        """
        modified_sequence = sequence[:]
        available_step_indices = list(range(len(sequence)))  # List of all step indices to try

        while available_step_indices:
            random_step_index = random.choice(available_step_indices)  # Choose a random step
            original_step = sequence[random_step_index]

            # Gather all used positions from the entire sequence
            used_positions = {t for step in sequence for t in step}

            # Generate a new step with the same number of boxes, ensuring adjacency and uniqueness
            new_step = []
            try:
                while len(new_step) < len(original_step):
                    # Get all available adjacent positions for the current step
                    available_positions = {
                        adj
                        for pos in original_step
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

                # Replace the chosen step with the new step
                modified_sequence[random_step_index] = new_step
                return modified_sequence  # Return immediately after successfully modifying a step

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
                       task_dur=300,
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

class SentenceReading(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'sentence_reading'

    def make_task_file(self,
                       run_number = None,
                       task_dur=30,
                       trial_dur=5.8,
                       iti_dur=0.2,
                       file_name=None,
                       stim_file=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / 'sentence_reading' / 'sentences_shuffled.csv')

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
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

class NonwordReading(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'nonword_reading'

    def make_task_file(self,
                       run_number = None,
                       task_dur=30,
                       trial_dur=5.8,
                       iti_dur=0.2,
                       file_name=None,
                       stim_file=None):

        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / 'nonword_reading' / 'nonwords_shuffled.csv')

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
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
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        stimuli = ['black_K'] * 12 + ['black_O'] * 12 + ['red_K'] * 3 + ['red_O'] * 3

        # shuffle stimuli
        random.shuffle(stimuli)

        t = 0

        for n in range(len(stimuli)):
            trial = {}
            trial['key_one'] = responses[0]
            trial['key_two'] = responses[1]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
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
                       responses = [1,2,3,4], # 1 = Key_one, 2 = Key_two, 3 = Key_three, 4 = Key_four
                       task_dur=300,
                       trial_dur=3.25,
                       iti_dur=0.5,
                       file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

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
                       file_name=None):

        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        for n in range(n_trials):
            trial = {}
            trial['key_one'] = responses[0]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
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
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

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


class FlexionExtension(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'flexion_extension'

    def make_task_file(self,
                       task_dur = 30,
                       trial_dur = 30,
                       iti_dur   = 0,
                       stim_dur = 2,
                       file_name = None):
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['stim'] = "flexion extension"
            trial['stim_dur'] = stim_dur
            trial['display_trial_feedback'] = False
            trial['trial_type'] = 'None'  # as there are no true or false responses
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

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
                       task_dur=300,
                       trial_dur=15,
                       sentence_dur=2,
                       file_name=None,
                       stim_file=None):

        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur)))
        trial_info = []
        t = 0

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / 'semantic_prediction' / 'semantic_prediction.csv')

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['key_true'] = responses[0]
            trial['key_false'] = responses[1]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['sentence_dur'] = sentence_dur
            trial['sentence'] = stim['sentence'][n]
            trial['trial_type'] = random.choice([0,1])
            last_word = [stim['wrong_word'][n], stim['right_word'][n]]
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
                       task_dur = 30,
                       trial_dur = 2,
                       iti_dur   = 0.5,
                       easy_prob=0.5,
                       file_name = None ):
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
            trial['num_stimuli'] = '4' if random.random() < easy_prob else '8'  # Randomly select difficulty
            trial['display_trial_feedback'] = True
            trial['feedback_type'] = 'acc'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur

            # Determine the number of stimuli to display based on trial difficulty
            num_stimuli = 4 if trial['num_stimuli'] == '4' else 8

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
                       task_dur=300,
                       trial_dur=6,
                       iti_dur=1.5,
                       file_name=None,
                       stim_file = None,
                       condition=None,
                       half=None):


        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1

        if stim_file:
            stim = pd.read_csv(self.stim_dir / self.name / stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}.csv')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]
            # Alternate between emotion and age conditions
            stim_emotion = stim[stim['condition'] == 'emotion']
            stim_age = stim[stim['condition'] == 'age']
            # Split each condition into halves
            first_half = zip(stim_emotion.iloc[:len(stim_emotion) // 2].iterrows(),
                             stim_age.iloc[len(stim_age) // 2:].iterrows())
            second_half = zip(stim_emotion.iloc[len(stim_emotion) // 2:].iterrows(),
                              stim_age.iloc[:len(stim_age) // 2].iterrows())
            stim = pd.concat([pd.concat([row1[1], row2[1]], axis=1).T for row1, row2 in itertools.chain(first_half, second_half)], ignore_index=True)

        if half: # Selects different stimuli for the social and control condition, to enable showing each story only once for each participant
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
            stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

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
            trial['stim'] = stim['picture'][n]
            trial['options'] = stim['options'][n]
            trial['condition'] = stim['condition'][n]
            trial['answer'] = stim['answer'][n]
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


class PictureSequence(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'picture_sequence'
        self.matching_stimuli = False # sequence of pictures are different for different conditions

    def generate_sequence(self):
        sequence = random.sample([1, 2, 3, 4], 4)
        return ' '.join(map(str, sequence))

    def make_task_file(self,
                       hand = 'right',
                       responses = [1,2,3,4], # 1 = Key_one, 2 = Key_two, 3 = Key_three, 4 = Key_four
                       run_number=None,
                       task_dur=30,
                       trial_dur=14,
                       iti_dur=1,
                       file_name=None,
                       stim_file = None,
                       condition=None):
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
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

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
            trial['condition'] = stim['condition'][n]
            trial['stim'] = stim['picture'][n]
            # choose random sequence
            trial['sequence'] = self.generate_sequence()
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class StorySequence(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'story_sequence'
        self.matching_stimuli = False # sequence of sentences are different for different conditions

    def generate_sequence(self):
        sequence = random.sample([1, 2, 3, 4], 4)
        return ' '.join(map(str, sequence))

    def make_task_file(self,
                       hand = 'right',
                       responses = [1,2,3,4], # 1 = Key_one, 2 = Key_two, 3 = Key_three, 4 = Key_four
                       run_number=None,
                       task_dur=30,
                       trial_dur=14,
                       iti_dur=1,
                       file_name=None,
                       stim_file = None,
                       condition=None):
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
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

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
            trial['condition'] = stim['condition'][n]
            trial['stim1'] = stim['Sentence1'][n]
            trial['stim2'] = stim['Sentence2'][n]
            trial['stim3'] = stim['Sentence3'][n]
            trial['stim4'] = stim['Sentence4'][n]
            # choose random sequence
            trial['sequence'] = self.generate_sequence()
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class ActionPrediction(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'action_prediction'
        self.matching_stimuli = False # sequence of pictures are different for different conditions

    def make_task_file(self, hand='right',
                       responses = [1,2],
                       run_number=None,
                       task_dur=300,
                       trial_dur=5,
                       iti_dur=1,
                       question_dur=4,
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
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}.csv', sep='\t')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

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
            trial['iti_dur'] = iti_dur
            trial['stim'] = stim['video'][n]
            trial['question'] = stim['question'][n]
            trial['options'] = stim['options'][n]
            trial['answer'] = stim['answer'][n]
            trial['condition'] = stim['condition'][n]
            trial['question_dur'] = question_dur
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
                       condition=None):

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
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        # Display a warning for romance clip 09 and up: The clips are repeated clips 1-8
        if run_number >= 9:
            Warning('Romance condition clips 9-10 are duplicates. They are the same as clips 1-8')

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            trial['stim'] = stim['video'][n]
            trial['condition'] = stim['condition'][n]
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info


class StrangeStories(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'strange_stories'
        self.matching_stimuli = True
        self.half_assigned = True

    def make_task_file(self,
                       hand='right',
                       responses = [1,2,3],
                       run_number = None,
                       task_dur=30,
                       trial_dur=30,
                       iti_dur=0,
                       answer_dur=5,
                       file_name=None,
                       stim_file=None,
                       condition=None,
                       half=None):

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
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        if half: # Selects different stimuli for the social and control condition, to enable showing each video only once for each participant (assign half the subjects one type of video as social and the other the other half of the videos as social)
            stim = stim[stim['half'] == half]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['key_one'] = responses[0]
            trial['key_two'] = responses[1]
            trial['key_three'] = responses[2]
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            trial['stim'] = stim['video'][n]
            trial['video_dur'] = stim['duration'][n]
            trial['answer_dur'] = answer_dur
            trial['question'] = stim['question'][n]
            trial['options'] = stim['options'][n]
            trial['condition'] = stim['condition'][n]
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
                       task_dur=300,
                       trial_dur=14,
                       iti_dur=1,
                       story_dur=10,
                       question1_dur=4,
                       file_name=None,
                       stim_file=None,
                       condition=None,
                       half=None):


        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}.csv')

        if condition:
            stim = stim[stim['condition'] == condition]
        else:
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        if half: # Selects different stimuli for the social and control condition, to enable showing each story only once for each participant
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



class FrithHappe(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'frith_happe'
        self.matching_stimuli = False

    def make_task_file(self,
                       hand='right',
                       responses = [1,2,3],
                       run_number = None,
                       task_dur=30,
                       trial_dur=28,
                       iti_dur=2,
                       question_dur=6,
                       file_name=None,
                       stim_file=None,
                       condition=None):

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
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['key_one'] = responses[0]
            trial['key_two'] = responses[1]
            trial['key_three'] = responses[2]
            if 'tom' in str(stim['condition'][n]):
                trial['trial_type'] = 2
            elif 'gd' in str(stim['condition'][n]):
                trial['trial_type'] = 3
            else:
                trial['trial_type'] = 1
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = True
            trial['stim'] = stim['video'][n]
            trial['video_dur'] = stim['duration'][n]
            trial['question_dur'] = question_dur
            trial['condition'] = stim['condition'][n]
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info



class Liking(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'liking'
        self.matching_stimuli = False

    def map_to_4point_scale(self, rating):
        """
        Map the liking rating from a 1-to-5 scale to the closest value on a 4-point scale
        (to be used in the scanner with the 4-button box).

        Parameters:
            rating (float): The rating on a 1-to-5 scale (can include decimals, since it's an average across online raters).
        Returns:
            int: The closest value on the 4-point scale (1, 2, 3, or 4).

        # Example usage:
        rating = 3.7
        closest_value = map_to_4point_scale(rating)
        print(f"The 1-to-5 rating {rating} maps closest to {closest_value} on the 4-point scale.")
        """
        if np.any((rating < 1) | (rating > 5)):
            raise ValueError("Rating must be between 1 and 5, inclusive.")

        # Normalize the rating to a 0-to-1 range
        normalized = (rating - 1) / 4
        # Map to the 4-point scale
        mapped_value = 1 + normalized * 3
        # Round to the nearest integer
        return round(mapped_value)


    def make_task_file(self,
                       hand='right',
                       responses = [1,2],
                       run_number = None,
                       task_dur=300,
                       trial_dur=28,
                       iti_dur=1,
                       question_dur=3,
                       file_name=None,
                       stim_file=None,
                       condition=None):

        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        if stim_file:
            stim = pd.read_csv(self.stim_dir / self.name / stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / self.name / f'{self.name}.csv')

        if condition:
            stim = stim[stim['condition'] == condition]
            # Randomize order with seed
            stim = stim.sample(frac=1, random_state=84).reset_index(drop=True)
        else:
            stim = stim.loc[~stim['condition'].str.contains('practice', na=False)]

        start_row = (run_number - 1) * n_trials
        end_row = run_number * n_trials - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['key_one'] = responses[0]
            trial['key_two'] = responses[1]
            trial['rating'] = int(stim['liking_effective'][n])
            trial['answer'] = stim['answer'][n]
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = True
            trial['stim'] = stim['video'][n]
            trial['video_dur'] = stim['duration'][n]
            trial['question_dur'] = question_dur
            trial['condition'] = stim['condition'][n]
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class Pong(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'pong'
        self.trajectories =  [
            (0.3, -0.15), (-0.3, -0.15), (0.1, -0.15), (-0.1, -0.15),
            (0.4, -0.15), (-0.4, -0.15), (0.6, -0.15), (-0.6, -0.15)
        ]

    def make_task_file(self,
                       hand = 'bimanual',
                       responses = [3,4], #3 = Key_three, 4 = Key_four
                       task_dur=30,
                       trial_dur=3.25,
                       iti_dur=0.5,
                       file_name=None,
                       run_number=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['key_left'] = responses[0]
            trial['key_right'] = responses[1]
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = True
            # choose random sequence
            trial['stim'] = random.choice(self.trajectories)
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            ut.dircheck(self.task_dir / self.name)
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
                       run_number=None,
                       hand='left',
                       responses=[3, 4]):

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


#%%
