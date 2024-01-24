# Create TaskFile file for different tasks
# March 2021: First version: Ladan Shahshahani  - Maedbh King - Suzanne Witt,
# Revised 2023: Bassel Arafat, Jorn Diedrichsen, Ince Hussain

import pandas as pd
import numpy as np
import random
import MultiTaskBattery.utils as ut



def shuffle_rows(dataframe):
    """
    randomly shuffles rows of the dataframe

    Args:
        dataframe (dataframe): dataframe to be shuffled
    Returns:
        dataframe (dataframe): shuffled dataframe
    """
    indx = np.arange(len(dataframe.index))
    np.random.shuffle(indx)
    dataframe = (dataframe.iloc[indx]).reset_index(drop = True)
    return dataframe

def add_start_end_times(dataframe, offset, task_dur):
    """
    adds start and end times to the dataframe

    Args:
        dataframe (dataframe): dataframe to be shuffled
        offset (float): offset of the task
        task_dur (float): duration of the task
    Returns:
        dataframe (dataframe): dataframe with start and end times
    """
    dataframe['start_time'] = np.arange(offset, offset + len(dataframe)*task_dur, task_dur)
    dataframe['end_time']   = dataframe['start_time'] + task_dur
    return dataframe

def make_run_file(task_list,
                  tfiles,
                  offset = 0,
                  instruction_dur = 5,
                  task_dur = 30):
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
    R = shuffle_rows(R)
    R = add_start_end_times(R, offset, task_dur+instruction_dur)
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
        self.exp_name   = const.exp_name
        self.task_dir = const.task_dir
        self.stim_dir   = const.stim_dir

class NBack(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'n_back'

    def make_task_file(self,
                        hand = 'right',
                        responses = [1,2], # 1 = match, 2 = no match
                        task_dur =  30,
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
                prev_stim[0]
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
                        task_dur =  30,
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
                        task_dur =  30,
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
                        task_dur =  30,
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

class RomanceMovie(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'romance_movie'

    def make_task_file(self,
                       run_number = None ,
                       task_dur=30,
                       trial_dur=30,
                       iti_dur=0,
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
            trial['stim'] = f'{run_number:02d}_romance.mov'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
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

    def make_task_file(self, hand='right',
                       responses = [1,2], # 1 = True, 2 = False
                       run_number=None,
                       task_dur=30,
                        trial_dur=14,
                        iti_dur=1, 
                        story_dur=10,
                        question_dur=4, file_name=None
                        , stim_file=None):

        # count number of trials
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0

        if stim_file:
            stim = pd.read_csv(stim_file)
        else:
            stim = pd.read_csv(self.stim_dir / 'theory_of_mind' / 'theory_of_mind.csv')

        start_row = (run_number - 1) * 2
        end_row = run_number * 2 - 1
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
                        'Brigand', 'Brocade', 'Casement',  'Cornice',\
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

    def get_adjacent_positions(self, pos, grid_size):
        x, y = pos
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1]:
                adjacent.append((new_x, new_y))
        return adjacent

    def generate_sequence(self, grid_size=(3, 4), sequence_length=4):
        sequence = [(random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))]

        while len(sequence) < sequence_length:
            possible_moves = set()
            for pos in sequence:
                for adj_pos in self.get_adjacent_positions(pos, grid_size):
                    if adj_pos not in sequence:
                        possible_moves.add(adj_pos)

            if not possible_moves:
                break  # Restart if no valid moves are possible

            sequence.append(random.choice(list(possible_moves)))

        return sequence

    def modify_sequence(self, sequence, grid_size=(3, 4)):
        def is_connected(seq):
            to_visit = {seq[0]}
            visited = set()
            while to_visit:
                pos = to_visit.pop()
                if pos in visited:
                    continue
                visited.add(pos)
                to_visit.update({adj for adj in self.get_adjacent_positions(pos, grid_size) if adj in seq})
            return len(visited) == len(seq)

        modified_sequence = sequence[:]  # Make a copy of the original sequence to modify
        while True:
            sequence_copy = modified_sequence[:]  # Work with a copy to avoid altering the original sequence during checks
            removed_position = sequence_copy.pop(0) if random.choice([True, False]) else sequence_copy.pop()

            # Generate a list of adjacent positions excluding the removed position and current sequence positions
            adjacent_positions = {adj_pos for pos in sequence_copy for adj_pos in self.get_adjacent_positions(pos, grid_size)}
            possible_moves = adjacent_positions - set(sequence_copy) - {removed_position}

            if possible_moves:
                sequence_copy.append(random.choice(list(possible_moves)))
                if is_connected(sequence_copy) and sequence_copy != sequence:
                    return sequence_copy  # Return the modified sequence if it's different and connected

    def make_task_file(self,
                        hand = 'right',
                        responses = [1,2], # 1 = Left, 2 = Right
                        task_dur=30,
                        trial_dur=7,
                        question_dur=3,
                        sequence_dur=4,
                        iti_dur=0.5,
                        grid_size=(3, 4),
                        sequence_length=8,
                        file_name=None):
        
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['key_left'] = responses[0]
            trial['key_right'] = responses[1]
            trial['correct_side'] = random.choice(['left', 'right'])
            if trial['correct_side'] == 'left':
                trial['trial_type'] = 0
            else:
                trial['trial_type'] = 1
            trial['trial_num'] = n
            trial['hand'] = hand
            original_sequence = self.generate_sequence(grid_size, sequence_length)
            trial['grid_sequence'] = original_sequence
            trial['modified_sequence'] = self.modify_sequence(original_sequence, grid_size)
            trial['display_trial_feedback'] = True
            trial['trial_dur'] = trial_dur
            trial['sequence_dur'] = sequence_dur
            trial['question_dur'] = question_dur
            trial['iti_dur'] = iti_dur
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

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
        
    def generate_sequence(self):
        sequence = [random.choice([1, 2, 3, 4])]
        while len(sequence) < 6:
            next_digit = random.choice([d for d in [1, 2, 3, 4] if d != sequence[-1]])
            sequence.append(next_digit)
        return ' '.join(map(str, sequence))

    def make_task_file(self,
                        hand = 'bimanual',
                        responses = [1,2,3,4], # 1 = Key_one, 2 = Key_two, 3 = Key_three, 4 = Key_four
                        task_dur=30,
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

class FlexionExtension(TaskFile):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'flexion_extension'

    def make_task_file(self,
                        task_dur =  30,
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




