# Create target file for different tasks
# @ Ladan Shahshahani  - Maedbh King - Suzanne Witt March 30 2021
from pathlib import Path
from itertools import count
import pandas as pd
import numpy as np
import os
import random
import glob
import re
import experiment_code.utils as ut



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
         'target_file':tfiles,
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

class Target():
    def __init__(self, const) :
        """ The Target class is class for creating target files for different tasks
        Args:
            const: module for constants
        """
        self.exp_name   = const.exp_name
        self.target_dir = const.target_dir

class NBack(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'n_back'

    def make_trial_file(self,
                        hand = 'right',
                        run_number = None,
                        task_dur =  30,
                        trial_dur = 2,
                        iti_dur   = 0.5,
                        stim = ['9.jpg','11.jpg','15.jpg','18.jpg','28.jpg'],
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
            # Determine if this should be N-2 repetition trial
            if n<2:
                trial['trial_type'] = 0
            else:
                trial['trial_type'] = np.random.randint(0,2)
            # Now choose the stimulus accordingly
            if trial['trial_type']==0:
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
            trial_info.to_csv(self.target_dir / self.name / file_name,sep='\t',index=False)
        return trial_info

class Rest(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'rest'

    def make_trial_file(self,
                        task_dur =  30,
                        file_name = None 
                        ,run_number = None):
        trial = {}
        trial['trial_num'] = [1]
        trial['trial_dur'] = [task_dur]
        trial['start_time'] = [0]
        trial['end_time'] =  [task_dur]
        trial_info = pd.DataFrame(trial)
        if file_name is not None:
            trial_info.to_csv(self.target_dir / self.name / file_name,sep='\t',index=False)
        return trial_info
    
class VerbGeneration(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'verb_generation'

    def make_trial_file(self,
                        hand = None,
                        task_dur =  30,
                        trial_dur = 2,
                        iti_dur   = 0.5,
                        file_name = None,
                        run_number = None ):
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        stim_path = Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'stimuli' / 'verb_generation'
        sitm_file = stim_path / 'verb_generation.csv'

        #shuffle the stimuli
        stim = pd.read_csv(sitm_file)
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
            trial['hand'] = hand
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
            ut.dircheck(self.target_dir / self.name)
            trial_info.to_csv(self.target_dir / self.name / file_name,sep='\t',index=False)

        return trial_info
    
class FlexionExtension(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'flexion_extension'

    def make_trial_file(self,
                        hand = None,
                        task_dur =  30,
                        trial_dur = 30,
                        iti_dur   = 0,
                        stim_dur = 2,
                        file_name = None,
                        run_number = None ):
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = 'None'  # as hand is not used
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
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)
        return trial_info
    
class TongueMovement(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'tongue_movement'

    def make_trial_file(self,
                        hand = None,
                        task_dur =  30,
                        trial_dur = 1,
                        iti_dur   = 0,
                        file_name = None,
                        run_number = None):
        n_trials = int(np.floor(task_dur / (trial_dur+iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = 'None'  # Hand is not used in this task
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
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)
        return trial_info

class AuditoryNarrative(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'auditory_narrative'

    def make_trial_file(self, run_number, task_dur=30, trial_dur=30, iti_dur=0, file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = 'None'  # Hand is not used in this task
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            # Select the appropriate audio file
            trial['stim'] = f'narrative_{run_number:02d}.wav'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)

            # Update for next trial:
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class RomanceMovie(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'romance_movie'

    def make_trial_file(self, run_number, task_dur=30, trial_dur=30, iti_dur=0, file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = 'None'  # Hand is not used in this task
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
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class SpatialNavigation(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'spatial_navigation'
        self.locations = ["KITCHEN", "BEDROOM", "FRONT-DOOR", "WASHROOM", "LIVING-ROOM"]

    def make_trial_file(self, run_number, task_dur=30, trial_dur=30, iti_dur=0, file_name=None):

        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        # Randomly select two different locations
        loc1, loc2 = random.sample(self.locations, 2)
        
        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = 'None'  # Hand is not used in this task
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
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)

        return trial_info

class TheoryOfMind(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'theory_of_mind'

    def make_trial_file(self, hand='right', run_number=None, task_dur=30, 
                        trial_dur=14, iti_dur=1, story_dur=10, 
                        question_dur=4, file_name=None):
        # Initialize necessary variables
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0


        stim_path = Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'stimuli' / 'theory_of_mind'
        stim_file = stim_path / 'theory_of_mind.csv'

         # Read and slice the stimuli based on run number
        stim = pd.read_csv(stim_file)
        start_row = (run_number - 1) * 2
        end_row = run_number * 2 - 1
        stim = stim.iloc[start_row:end_row + 1].reset_index(drop=True)

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = hand
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['story'] = stim['story'][n] 
            trial['question'] = stim['question'][n]  
            trial['condition'] = stim['condition'][n]
            trial['response'] = stim['response'][n]
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
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)
        return trial_info

class DegradedPassage(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'degraded_passage'

    def make_trial_file(self, run_number, task_dur=30, trial_dur=15, iti_dur=0, file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = 'None'  # Hand is not used in this task
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
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)

        return trial_info
    
class ActionObservation(Target):
    def __init__(self, const):
        super().__init__(const)
        self.name = 'action_observation'
        self.knot_names = ['Abyss', 'Adage', 'Ampere', 'Arbor', 'Baron', 'Belfry', 'Bramble',\
                        'Brigand', 'Brocade', 'Casement', 'Chamois', 'Coffer', 'Cornice',\
                        'Farthing', 'Fissure', 'Flora', 'Frontage', 'Gadfly', 'Garret', \
                        'Gentry', 'Henchman', 'Magnate', 'Mutton', 'Perry', 'Phial', \
                        'Placard', 'Polka', 'Purser', 'Rosin', 'Shilling', 'Simper', \
                        'Spangle', 'Squire', 'Vestment', 'Wampum', 'Wicket']



    def make_trial_file(self, run_number, task_dur=30, trial_dur=14, iti_dur=1, file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []

        t = 0

        for n in range(n_trials):
            trial = {}
            trial['trial_num'] = n
            trial['hand'] = 'None'  # Hand is not used in this task
            trial['trial_dur'] = trial_dur
            trial['iti_dur'] = iti_dur
            trial['display_trial_feedback'] = False
            knot_index = (run_number - 1)
            if n == 0:
                trial['stim'] = f'knotAction{self.knot_names[knot_index]}.mov'
            else:
                trial['stim'] = f'knotControl{self.knot_names[knot_index]}.mov'
            trial['start_time'] = t
            trial['end_time'] = t + trial_dur + iti_dur
            trial_info.append(trial)
            t = trial['end_time']

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.target_dir / self.name / file_name, sep='\t', index=False)

        return trial_info
### ====================================================================================================
# What follows is potentially depreciated code, which I think is unecessarily complicated
### ====================================================================================================



class Session():

    def __init__(self, study_name,
                 task_list = ['visual_search', 'action_observation_knots', 'flexion_extension',
                              'finger_sequence', 'theory_of_mind', 'n_back', 'semantic_prediction',
                              'rest'],
                 instruct_dur = 5, task_dur = 30, num_runs = 8,
                 tile_runs = 1, counter_balance = True,
                 session = 1, start_hand = 'right'):

        self.study_name      = study_name      # 'fmri' or 'behavioral'
        self.task_list       = task_list       # list of tasks. Default is the list for pontine project
        self.instruct_dur    = instruct_dur    # instruction period
        self.task_dur        = task_dur        # duration of each task
        self.num_runs        = num_runs        # number of runs
        self.tile_runs       = tile_runs       #
        self.counter_balance = counter_balance # counter balance runs? default: True
        self.session         = session         # session number
        self.start_hand      = start_hand      # starting hand of the session

        self.hands_list = ['right', 'left']

    def _check_task_run(self):
        """
        randomly picks a target file for the current run
        """
        # check if task exists in dict
        exists_in_dict = [True for key in self.target_dict.keys() if self.task_name==key]
        if not exists_in_dict:
            self.target_dict.update({self.task_name: self.fpaths})

        # create run dataframe
        # random.seed(self.task_num+1)
        target_files_sample = [self.target_dict[self.task_name].pop(random.randrange(len(self.target_dict[self.task_name]))) for _ in np.arange(self.tile_runs)]
        # target_files_sample = [self.target_dict[self.task_name] for _ in np.arange(self.tile_runs)]


        return target_files_sample

    def _test_counterbalance(self):
        """
        Testing whether tasks are counter balanced
        """
        filenames = sorted(glob.glob(os.path.join(consts.run_dir, self.study_name, '*run_*')))

        dataframe_all = pd.DataFrame()
        for i, file in enumerate(filenames):
            dataframe = pd.read_csv(file)
            dataframe['run'] = i + 1
            dataframe['task_num_unique'] = np.arange(len(dataframe)) + 1
            dataframe_all = pd.concat([dataframe_all, dataframe])

        # create new column
        dataframe_all['block_name_unique'] = dataframe_all['task_name'] + '_' + dataframe_all['task_iter'].astype(str)
        # dataframe_all['task_name_unique'] = dataframe_all['task_name']

        task = np.array(list(map({}.setdefault, dataframe_all['block_name_unique'], count()))) + 1
        last_task = list(task[0:-1])
        last_task.insert(0,0)
        last_task = np.array(last_task)
        last_task[dataframe_all['task_num_unique']==1] = 0

        dataframe_all['last_task'] = last_task
        dataframe_all['task']      = task
        dataframe_all['task_num']  = task

        # get pivot table
        f = pd.pivot_table(dataframe_all, index=['task'], columns=['last_task'], values=['task_num'], aggfunc=len)

        return sum([sum(f['task_num'][col]>5) for col in f['task_num'].columns])

    def _counterbalance_runs(self):
        """
        checks if the tasks are counter balanced. If not, creates new run file
        """
        while self._test_counterbalance() > 0:
            print('not balanced ...')

            # delete any run files that exist in the folder
            files = glob.glob(os.path.join(consts.run_dir, self.study_name, '*run*.tsv'))
            # for f in files:
            #     os.remove(f)
            self.make_run_files()

        print('these runs are perfectly balanced')

    def make_target_files(self):
        """
        makes target files for all the tasks in task list
        """
        for self.task_name in self.task_list:
            # get the directory for the target file
            target_dir = os.path.join(consts.target_dir, self.study_name, self.task_name)

            # delete any target files that exist in the folder
            files = glob.glob(os.path.join(target_dir, '*.tsv'))
            # for f in files:
            #     os.remove(f)

            # create an array for run numbers
            runs = range((self.session - 1)*self.num_runs, (self.session - 1)*self.num_runs+self.num_runs)

            for run in runs:
                # make target files
                TaskClass = TASK_MAP[self.task_name]
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # need to figure out a way to input task parameters flexibly
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # determine the hand assigned for runs of the session
                if run < (self.session - 1)*self.num_runs + int(self.num_runs/2):
                    hand = self.start_hand
                else:
                    hand = [x for x in self.hands_list if x != self.start_hand][0]

                Task_target = TaskClass(run_number = run, study_name = self.study_name, hand = hand)
                Task_target._make_files()

    def make_run_files(self):
        """
        makes run file
        """
        # create run files
        self.target_dict = {}

        # create an array for run numbers
        runs = range((self.session - 1)*self.num_runs, (self.session - 1)*self.num_runs+self.num_runs)

        for run in runs:
            self.cum_time = 0.0
            self.all_data = []

            for self.task_num, self.task_name in enumerate(self.task_list):

                # get target files for `task_name`
                self.task_target_dir = os.path.join(consts.target_dir, self.study_name, self.task_name)
                self.fpaths = sorted(glob.glob(os.path.join(self.task_target_dir, f'*{self.task_name}_{self.task_dur}sec_*.tsv')))
                target_file = self.fpaths[run]
                iter = 0
                dataframe = pd.read_csv(target_file)

                start_time = dataframe.iloc[0]['start_time'] + self.cum_time
                end_time   = dataframe.iloc[-1]['start_time'] + dataframe.iloc[-1]['trial_dur'] + self.instruct_dur + self.cum_time

                target_file_name = Path(target_file).name
                num_sec = re.findall(r'\d+(?=sec)', target_file)[0]
                target_num = re.findall(r'\d+(?=.tsv)', target_file)[0]
                num_trials = len(dataframe)

                data = {'task_name': self.task_name, 'task_iter': iter + 1, 'task_num': self.task_num + 1, # 'block_iter': iter+1
                        'num_trials': num_trials, 'target_num': target_num, 'num_sec': num_sec,
                        'target_file': target_file_name, 'start_time': start_time, 'end_time': end_time,
                        'instruct_dur': self.instruct_dur}

                self.all_data.append(data)
                self.cum_time = end_time
                # ---------------------------------------------------------------------------------------

            # shuffle order of tasks within run
            df_run = pd.DataFrame.from_dict(self.all_data)
            df_run = df_run.sample(n=len(df_run), replace=False)

            # correct `start_time`, `run_time`
            df_run['start_time'] = sorted(df_run['start_time'])
            df_run['end_time'] = sorted(df_run['end_time'])

            timestamps = (np.cumsum(df_run['num_sec'].astype(int) + df_run['instruct_dur'].astype(int))).to_list()
            df_run['end_time'] = timestamps
            timestamps.insert(0, 0)
            df_run['start_time'] = timestamps[:-1]

            df_run.drop({'num_sec'}, inplace=True, axis=1)

            # save run file
            run_name = 'run_' +  f'{run+1:02d}' + '.tsv'
            filepath = os.path.join(consts.run_dir, self.study_name)
            consts.dircheck(filepath)
            df_run.to_csv(os.path.join(filepath, run_name), index=False, header=True)

    def check_counter_balance(self):

        if self.counter_balance:
            self._counterbalance_runs()
        else:
            print(f"You have chosen not to counter balance runs!")


# define classes for each task target file.
# add your tasks as classes
class FingerSequence(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 3.25,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = True,
                 task_dur=30, tr = 1, seq_length = 6):

        super(FingerSequence, self).__init__(study_name = study_name, task_name = 'finger_sequence', hand = hand,
                                             trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number,
                                             display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)
        self.seq_length = seq_length # number of digits to be pressed in a trial

        self.feedback_type  = 'acc'
        self.trials_info = {"condition_name": ["simple", "complex"], "trial_type": [None]}
        # the sequences
        self.seq = {}
        ## EXperimental:
        self.seq['complex'] = ['2 4 3 5 4 3', '3 2 4 5 4 2',
                        '4 3 5 2 5 3', '5 2 3 4 5 2']
        ## Control
        self.seq['simple'] = ['2 2 2 2 2 2', '3 3 3 3 3 3',
                        '4 4 4 4 4 4', '5 5 5 5 5 5']

    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        n_complex = int(self.num_trials/2)
        n_simple  = self.num_trials - n_complex

        # shuffle sequences for complex and simple
        np.random.shuffle(self.seq['complex'])
        np.random.shuffle(self.seq['simple'])

        # self.target_dict['trial_type'] = ['None' for trial_number in range(self.num_trials)]
        self.target_dataframe['trial_type'] = ['None' for trial_number in range(self.num_trials)]

        # fill in fields ?????????????????????????
        # ## randomize order of complex and simple conditions across runs
        # if self.run_number%2 == 0: # in even runs complex sequences come first
        #     self.target_dict['condition_name'] = np.concatenate((np.tile('complex', n_complex), np.tile('simple', n_complex)), axis=0)
        #     self.target_dict['sequence']       = np.concatenate((self.seq['complex'], self.seq['simple']), axis=0).T.flatten()
        # else: # in odd runs simple sequences come first
        #     self.target_dict['condition_name'] = np.concatenate((np.tile('simple', n_simple), np.tile('complex', n_simple)), axis=0)
        #     self.target_dict['sequence']       = np.concatenate((self.seq['simple'], self.seq['complex']), axis=0).T.flatten()

        self.target_dataframe['condition_name'] = np.concatenate((np.tile('complex', n_complex), np.tile('simple', n_complex)), axis=0)
        self.target_dataframe['sequence']       = np.concatenate((self.seq['complex'], self.seq['simple']), axis=0).T.flatten()
        self.target_dataframe['feedback_type']  = [self.feedback_type for trial_number in range(self.num_trials)]

        # randomly shuffle rows of the dataframe
        dataframe = self.shuffle_rows(self.target_dataframe)

        return dataframe

    def _make_files(self):
        """
        makes target file and (if exists) related task info and  saves them
        """

        # save target file
        self.df = self._add_task_info(random_state=self.run_number)
        self.df = self.make_trials_time(self.df)
        self.save_target_file(self.df)

class SternbergOrder(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 6,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = True,
                 task_dur=30, tr = 1, digit_dur = 0.75, delay_dur = 0.5,
                 prob_dur = 1, load = 6):
        super(SternbergOrder, self).__init__(study_name = study_name, task_name = 'sternberg_order', hand = hand,
                                             trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number,
                                             display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)

        self.trials_info   = {'condition_name':["forward"], "trial_type":[True, False]}
        self.feedback_type = 'acc'
        self.prob_dur      = prob_dur  # length of time the prob remains on the screen (also time length for response to be made)
        self.digit_dur     = digit_dur # length of time each digit remains on the screen
        self.delay_dur     = delay_dur # duration of the delay
        self.load          = load      # memory load

    def _get_stim_digits(self):

        # generate random numbers betweem 1 and 9
        rand_nums = [np.random.choice(range(1, 10), size = 6, replace = False) for i in range(self.num_trials)]
        ## convert the random numbers to str and concatenate them
        self.stim_str = []
        for nums in rand_nums:
            rand_str = ""
            for x in nums: rand_str += str(x) + " "
            self.stim_str.append(rand_str)

    def _get_prob_digits(self):
        # determine the prob stim for each trial based on trial type
        self.prob_stim = []
        for trial in range(self.num_trials):
            # get the trial_type for the current trial
            current_tt = self.target_dataframe['trial_type'].loc[trial]

            # get the current stims
            current_stim = self.target_dataframe['stim'].loc[trial]
            current_stim_digits = current_stim.split()

            # pick two random digits from current stimulus
            probs_str = np.random.choice(current_stim_digits, size = 2, replace = False)
            # determine the order
            prob_order = [current_stim_digits.index(x) for x in probs_str]

            if ~ current_tt: # if it's False:
                # the trial is false so the order will be changed
                ## find the one that comes last and put it as the first digit in the prob
                first_prob_digit = current_stim_digits[max(prob_order)]
                last_prob_digit  = current_stim_digits[min(prob_order)]
            else: # if it's True
                ## find the one that comes first and put it as the first digit in the prob
                first_prob_digit = current_stim_digits[min(prob_order)]
                last_prob_digit  = current_stim_digits[max(prob_order)]

            # generate the prob stim and append it
            self.prob_stim.append(first_prob_digit + " " + last_prob_digit)

    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        # randomly select trialTypes
        # self.target_dict['trial_type'] = np.random.choice([True, False], size = self.num_trials, replace=True)
        self.target_dataframe['trial_type'] = (int(self.num_trials/len(self.trials_info['trial_type'])))*self.trials_info['trial_type']

        self.target_dataframe['delay_dur'] = [self.delay_dur for i in range(self.num_trials)]
        self.target_dataframe['digit_dur'] = [self.digit_dur for i in range(self.num_trials)]
        self.target_dataframe['prob_dur']  = [self.prob_dur for i in range(self.num_trials)]

        # get random digits as stimulus
        self._get_stim_digits()

        self.target_dataframe['stim'] = self.stim_str

        # get the prob stimulus of trials
        self._get_prob_digits()

        self.target_dataframe['prob_stim'] = self.prob_stim

        dataframe = self.shuffle_rows(self.target_dataframe)

        return dataframe

    def _make_files(self):
        """
        makes target file and (if exists) related task info and  saves them
        """

        # save target file
        self.df = self._add_task_info(random_state=self.run_number)
        self.df = self.make_trials_time(self.df)
        self.save_target_file(self.df)

class VisuospatialOrder(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 6,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = True,
                 task_dur=30, tr = 1, dot_dur = 0.75, delay_dur = 0.5,
                 prob_dur = 1, load = 6, min_distance = 1, width = 8):
        super(VisuospatialOrder, self).__init__(study_name = study_name, task_name = 'visuospatial_order', hand = hand,
                                                trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number,
                                                display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)


        self.trials_info = {'condition_name':[None], "trial_type":[True, False]}
        self.prob_dur     = prob_dur     # length of time the prob remains on the screen (also time length for response to be made)
        self.digit_dur    = dot_dur      # length of time each digit remains on the screen
        self.delay_dur    = delay_dur    # duration of the delay
        self.load         = load         # memory load
        self.width        = width        # width of the enclosing square (or can be circle)
        self.min_distance = min_distance # minimum distance between dots

    def _get_trial_stim_coords(self):
        """
        creates coordinates for dots to be used as stimulus
        """
        # generate random points
        x = np.random.uniform(-self.width/2, self.width/2)
        y = np.random.uniform(-self.width/2, self.width/2)

        # checks the distance between all the points to make sure they are at a minimum distance
        counter = 2 # counter for the point
        self.dot_xyz = []
        self.dot_xyz.append([x, y])
        while counter < self.load + 1:
            # start generating the other points
            ## generate another random point
            next_point = False
            xi = np.random.uniform(-self.width/2, self.width/2)
            yi = np.random.uniform(-self.width/2, self.width/2)

            # check the distance between this point and all the other points already in the list dot_xys
            for point in self.dot_xyz:
                # calculate the distance
                distance = np.sqrt(((point[0] - xi)**2) + ((point[1] - yi)**2))
                # print(distance)

                # if the distance is lower than a threshold, break the loop and generate another point
                if distance < self.min_distance:
                    next_point = True
                    break
                else:
                    continue
            # append the point to the list of dots only if its distance from
            # all the other points is larger than min_distance
            if not next_point:
                counter+=1
                self.dot_xyz.append([xi, yi])

    def _get_trial_prob_coords(self, trial_number):
        """
        create coordinates for the prob dots based on the trial type of the current trial
        """

        # randomly pick two of the dots for probe based on the trial type
        # get the trial_type for the current trial
        current_tt = self.target_dataframe['trial_type'].loc[trial_number]

        # pick two dots
        rand_probs = np.random.choice(len(self.dot_xyz), size = 2, replace = False)

        if ~ current_tt: # False trial
            # the trial is false so two wrong digits with wrong order can be generated
            # sort the indices in descending order to make sure that their order is flipped
            self.probs_idx = np.sort(rand_probs)[::-1]
        else: # True trial
            # sort the indices in ascending order to make sure that their order is conserved
            self.probs_idx = np.sort(rand_probs)

        self.probs_xyz  = [self.dot_xyz[i] for i in self.probs_idx]

    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        # ---------------------------------------------------------------
        # # 1. assign trial types
        # n_trials_T = int(self.num_trials/2) # ??????????
        # n_trials_F = self.num_trials - n_trials_T

        # trials_True  = np.tile(True, n_trials_T)
        # trials_False = np.tile(False, n_trials_F)

        # trial_types = np.concatenate((trials_True, trials_False), axis = 0)

        # ## now randomly shuffle trials
        # np.random.shuffle(trial_types)
        # self.target_dict['trial_type'] = trial_types
        # ----------------------------------------------------------------
        # or 2.
        # self.target_dict['trial_type'] = np.random.choice([True, False], size = self.num_trials, replace=True)
        self.target_dataframe['trial_type'] = (int(self.num_trials/len(self.trials_info['trial_type'])))*self.trials_info['trial_type']
        # ------------------------------------------------------------------

        self.target_dataframe['delay_dur'] = [self.delay_dur for i in range(self.num_trials)]
        self.target_dataframe['dot_dur']   = [self.digit_dur for i in range(self.num_trials)]
        self.target_dataframe['prob_dur']  = [self.prob_dur for i in range(self.num_trials)]
        self.target_dataframe['width']     = [self.width for i in range(self.num_trials)]

        # creating empty lists for coordinates of dots
        coords_stim = []
        coords_prob = []
        # loop over trials and create coordinates of stim and prob
        for trial_number in range(self.num_trials):

            # get coordinates for stimulus dot
            self._get_trial_stim_coords()
            coords_stim.append(self.dot_xyz)

            # get coordinates for prob stimulus
            self._get_trial_prob_coords(trial_number)
            coords_prob.append(self.probs_xyz)

        self.target_dataframe['xyz_stim'] = pd.Series(coords_stim)
        self.target_dataframe['xyz_prob'] = pd.Series(coords_prob)
        # randomly shuffle rows of the dataframe
        dataframe = self.shuffle_rows(self.target_dataframe)

        return dataframe

    def _make_files(self):
        """
        makes target file and (if exists) related task info and  saves them
        """

        # save target file
        self.df = self._add_task_info(random_state=self.run_number)
        self.df = self.make_trials_time(self.df)
        self.save_target_file(self.df)

class VisualSearch(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right',
                 trial_dur = 2, iti_dur = 0.5, run_number = 1, display_trial_feedback = True,
                 task_dur = 30, tr = 1, replace = False):

        super(VisualSearch, self).__init__(study_name = study_name, task_name = 'visual_search', hand = hand,
                                           trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number,
                                           display_trial_feedback = display_trial_feedback, task_dur = task_dur,
                                           tr = tr)

        self.feedback_type  = 'acc'
        # self.instruct_dur = 5
        # self.replace = False
        self.orientations   = list([90, 180, 270, 360])
        self.trials_info = {'condition_name': {'easy': 4, 'hard': 8}, 'trial_type': [True, False]}

    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        # get `num_stims`
        self.num_stims = int(self.num_trials / len(self.trials_info['condition_name']))

        conds       = [self.trials_info['condition_name'][key] for key in self.trials_info['condition_name'].keys()]
        conds_names = [key for key in self.trials_info['condition_name'].keys()]

        self.target_dataframe['stim']            = (int(self.num_trials/len(conds)))*conds
        self.target_dataframe['condition_name']  = (int(self.num_trials/len(conds_names)))*conds_names
        # randomly shuffle true falses before assigning it to stim
        trial_type_list = (int(self.num_trials/len(self.trials_info['trial_type'])))*self.trials_info['trial_type']
        random.shuffle(trial_type_list)
        self.target_dataframe['trial_type']      = trial_type_list

        self.target_dataframe['feedback_type']   = [self.feedback_type for trial_number in range(self.num_trials)]


        # dataframe['trial_type'] = dataframe['trial_type'].sort_values().reset_index(drop=True)
        self.target_dataframe['stim'] = self.target_dataframe['stim'].astype(int) # convert stim to int

        # randomly shuffle rows of the dataframe
        dataframe = self.shuffle_rows(self.target_dataframe)

        return dataframe

    def _make_search_display(self, display_size, orientations, trial_type):
        # make location and orientations lists (for target and distractor items)

        # STIM POSITIONS
        grid_h_dva = 8.4
        grid_v_dva = 11.7

        n_h_items = 6
        n_v_items = 8

        item_h_pos = np.linspace(-grid_h_dva / 2.0, +grid_h_dva/ 2.0, n_h_items)
        item_v_pos = np.linspace(-grid_v_dva / 2.0, +grid_v_dva / 2.0, n_v_items)

        grid_pos = []
        for curr_h_pos in item_h_pos:
            for curr_v_pos in item_v_pos:
                grid_pos.append([curr_h_pos, curr_v_pos])

        locations = random.sample(grid_pos, display_size)

        ## STIM ORIENTATIONS
        orientations_list = orientations*int(display_size/4)

        # if trial type is false - randomly replace target stim (90)
        # with a distractor
        if not trial_type:
            orientations_list = [random.sample(orientations[1:],1)[0] if x==90 else x for x in orientations_list]

        # if trial is true and larger than 4, leave one target stim (90) in list
        # and randomly replace the others with distractor stims
        if display_size >4 and trial_type:
            indices = [i for i, x in enumerate(orientations_list) if x == 90]
            indices.pop(0)
            new_num = random.sample(orientations[1:],2) # always assumes that orientations_list is as follows: [90,180,270,360]
            for i, n in zip(*(indices, new_num)):
                orientations_list[i] = n

        return dict(enumerate(locations)), dict(enumerate(orientations_list))

    def _save_visual_display(self, dataframe):
        # add visual display cols
        display_pos, orientations_correct = zip(*[self._make_search_display(cond, self.orientations, trial_type) for (cond, trial_type) in zip(dataframe["stim"], dataframe["trial_type"])])

        data_dicts = []
        for trial_idx, trial_conditions in enumerate(display_pos):
            for condition, point in trial_conditions.items():
                data_dicts.append({'trial': trial_idx, 'stim': condition, 'xpos': point[0], 'ypos': point[1], 'orientation': orientations_correct[trial_idx][condition]})

        # save out to dataframe
        df_display = pd.DataFrame.from_records(data_dicts)

        # save out visual display
        str_part = self.target_filename.partition(self.task_name)
        visual_display_name = 'display_pos' + str_part[2]+".tsv"
        # self.target_dir = consts.target_dir / self.study_name / self.task_name
        df_display.to_csv(os.path.join(self.target_dir, visual_display_name))

    def _make_files(self):
        """
        makes target file and (if exists) related task info and  saves them
        """

        # save target file
        self.df = self._add_task_info(random_state=self.run_number)
        self.df = self.make_trials_time(self.df)
        self.save_target_file(self.df)

        # save info for the visual display
        self._save_visual_display(self.df)

class SemanticPrediction(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 6,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = True,
                 task_dur=30, tr = 1, stem_word_dur = 0.5, last_word_dur = 1.5, frac = 0.3):
        super(SemanticPrediction, self).__init__(study_name = study_name, task_name = 'semantic_prediction', hand = hand,
                                                 trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number,
                                                 display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)

        self.trials_info = {"condition_name": {"high cloze": "easy", "low cloze": "hard"},
                            "CoRT_descript": ["strong non-CoRT", "ambiguous", "strong CoRT"]}

        self.feedback_type = 'acc'
        self.stem_word_dur = stem_word_dur  # length of time the prob remains on the screen (also time length for response to be made)
        self.last_word_dur = last_word_dur  # length of time each digit remains on the screen
        self.frac          = frac           # ??????

    def _get_stim(self):
        """
        get stimulus dataframe
        """
        # read in the stimulus csv file
        # stim_dir = os.path.join(consts.stim_dir, self.study_name, self.task_name)
        stim_dir = os.path.join(consts.stim_dir, self.task_name)
        stim_df  = pd.read_csv(os.path.join(stim_dir, 'sentence_validation.tsv'))

        self.log_df = pd.read_csv(os.path.join(stim_dir, f'semantic_prediction_logging_{self.run_number}.tsv'))

        # conds = [self.balance_blocks['condition_name'][key] for key in self.balance_blocks['condition_name'].keys()]
        conds   = list(self.trials_info['condition_name'].keys())
        self.stim_df = stim_df.query(f'cloze_descript=={conds}')
        # use stimuli not already extracted
        ## first get the cloze_descript
        descript = self.log_df.query(f'cloze_descript=={conds}')
        not_extracted = descript["extracted"] != "TRUE"
        self.stim_df = self.stim_df.loc[not_extracted.values]

        # get full sentence:
        sentence = self.stim_df['full_sentence']
        # strip erroneous characters from sentences
        self.stim_df['stim'] = sentence.str.replace('|', ' ')

    def _balance_design(self, random_state):

        # group the dataframe according to `balance_blocks`
        self.stim_df = self.stim_df.groupby([*self.trials_info], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=random_state, replace=False)).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        num_stims = int(self.num_trials / len(self.trials_info['condition_name']))
        self.stim_df = self.stim_df.groupby('condition_name', as_index=False).apply(lambda x: x.sample(n=num_stims, random_state=random_state, replace=False)).reset_index(drop=True)

    def _add_random_word(self, random_state, columns):
        """ sample `frac_random` and add to `full_sentence`
            Args:
                dataframe (pandas dataframe): dataframe
            Returns:
                dataframe with modified `full_sentence` col
        """

        # group stimuli and sample
        group_columns = self.stim_df.groupby(columns)
        sample_group = group_columns.apply(lambda x: x.sample(frac=self.frac, replace=False, random_state=random_state))
        # find the extracted rows of the stimuli df
        extracted = self.log_df["full_sentence"].isin(sample_group["full_sentence"])

        # load in the logging file and update the extracted column
        self.log_df["extracted"] = extracted.values

        ## save the new logging
        log_dir = os.path.join(consts.stim_dir, self.task_name)
        self.log_df.to_csv(os.path.join(log_dir, f'semantic_prediction_logging_{self.run_number+1}.tsv'), index=False)

        idx = sample_group.index
        sampidx = idx.get_level_values(len(columns)) # get third level
        self.stim_df["trial_type"] = ~self.stim_df.index.isin(sampidx)
        self.stim_df["last_word"]  = self.stim_df.apply(lambda x: x["random_word"] if not x["trial_type"] else x["target_word"], axis=1)

        # shuffle the order of the trials
        self.target_df = self.stim_df.apply(lambda x: x.sample(n=self.num_trials, random_state=random_state, replace=False)).reset_index(drop=True)
    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        # get `num_stims`
        self.num_stims = int(self.num_trials / len(self.trials_info['condition_name']))

        # get the stimuls dataframe
        self._get_stim()

        self.stim_df['condition_name']         = self.stim_df['cloze_descript'].apply(lambda x: self.trials_info['condition_name'][x])
        self.stim_df['stem_word_dur']          = self.stem_word_dur
        self.stim_df['last_word_dur']          = self.last_word_dur
        self.stim_df['trial_dur_correct']      = (self.stim_df['word_count'] * self.stim_df['stem_word_dur']) + self.iti_dur + self.stim_df['last_word_dur']
        self.stim_df['display_trial_feedback'] = self.display_trial_feedback
        self.stim_df['replace_stimuli']        = self.replace
        self.stim_df['feedback_type']          = self.feedback_type

        # self.stim_df.drop({'full_sentence'}, inplace=True, axis=1)

        # balance design
        self._balance_design(random_state=random_state)

        # add random word based on `self.frac`. target_dataframe is created after using this method
        self._add_random_word(random_state=random_state, columns=['condition_name']) # 'CoRT_descript'

        self.target_dataframe = pd.concat([self.target_dataframe, self.target_df], axis = 1)

        # randomly shuffle rows of the dataframe
        dataframe = self.shuffle_rows(self.target_dataframe)

        return dataframe

    def _make_files(self):
        """
        makes target file and (if exists) related task info and  saves them
        """

        # save target file
        self.df = self._add_task_info(random_state=self.run_number)
        self.df = self.make_trials_time(self.df)
        self.save_target_file(self.df)

class ActionObservationKnots(Target):
    def __init__(self, study_name = 'behavioral', hand = None, trial_dur = 14,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = False,
                 task_dur=30, tr = 1):
        super(ActionObservationKnots, self).__init__(study_name = study_name, task_name = 'action_observation_knots', hand = None,
                                           trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number,
                                           display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)

        self.trials_info = {'condition_name': ['knot'], 'session_list': [1,2]}

    def _get_video(self):
        """
        get the video filenames in a dataframe
        """

        # load in stimuli
        # stim_dir = os.path.join(consts.stim_dir, self.study_name, self.task_name)
        stim_dir = os.path.join(consts.stim_dir, self.task_name)
        stim_df  = pd.read_csv(os.path.join(stim_dir, 'action_observation_knots.tsv'))

        # remove all filenames where any of the videos have not been extracted
        stims_to_remove = stim_df.query('extracted==False')["video_name_action"].to_list()
        self.stim_df    = stim_df[~stim_df["video_name_action"].isin(stims_to_remove)]

    def _balance_design(self, random_state):
        self.stim_df = self.stim_df.groupby([*self.trials_info], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=random_state, replace=self.replace)).reset_index(drop=True)
        # ensure that only `num_trials` are sampled
        self.stim_df = self.stim_df.sample(n=int(self.num_trials/2), random_state=random_state, replace=False).reset_index(drop=True)

    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        # get `num_stims`
        self.num_stims = int(self.num_trials / len(self.trials_info['condition_name']))

        # get stimulus dataframe
        self._get_video()

        types = ['action', 'control']
        self.target_dataframe['trial_type'] = types *int(self.num_trials/2) # there are no true of false responses
        self.target_dataframe['hand']       = ['None' for i in range(self.num_trials)] # hand is not used

        self.stim_df['stim_action']  = self.stim_df['video_name_action'] + '.mov'
        self.stim_df['stim_control'] = self.stim_df['video_name_control'] + '.mov'

        self.stim_df.drop({'video_name_action', 'video_name_control'}, inplace=True, axis=1)

        self._balance_design(random_state)
        # make target dataframe
        stim_action = self.stim_df['stim_action'].values[0]
        stim_control = self.stim_df['stim_control'].values[0]
        self.stim_df.drop({'stim_action', 'stim_control'}, inplace=True, axis=1)
        self.target_dataframe['stim'] = [stim_action, stim_control]

        # randomly shuffle rows of the dataframe/ no need to shuffle rows. Action always first, control second
        # dataframe = self.shuffle_rows(self.target_dataframe)

        return self.target_dataframe

    def _make_files(self):
        """
        makes target file and (if exists) related task info and  saves them
        """

        # save target file
        self.df = self._add_task_info(random_state=self.run_number)
        self.df = self.make_trials_time(self.df)
        self.save_target_file(self.df)


class SocialPrediction(Target):
    pass

# Functions to do the job
def make_task_target(task_name = 'visual_search', study_name = 'behavioral', hand = 'right'):
    """
    creates target file for a specific task.
    can be used for runs of individual tasks and also for testing one task
    Args:
        task_name - name of the task
        study_name - either 'behavioral' or 'fmri'
        hand - hand used for the task
    """

    # get the task class
    TaskClass = TASK_MAP[task_name]

    # make target file
    Task_target = TaskClass(run_number = 1, study_name = study_name, hand = hand)
    Task_target._make_files()

    return


def make_files(task_list, study_name = 'behavioral', num_runs = 8,
               start_hand = 'right', session = 1):
    """
    make target files and run files
    Args:
        study_name - either 'fmri' or 'behavioral'
        num_runs   - number of runs that you want to create
        start_hand - starting hand of the session
    """
    Sess = Session(task_list = task_list, study_name = study_name,
                   num_runs = num_runs, start_hand = start_hand,
                   session = session)
    Sess.make_target_files()
    Sess.make_run_files()
    Sess.check_counter_balance()

    return


if __name__ == "__main__":
    # Example code
    ## behavioral

    ## fmri
    make_files(study_name='fmri', num_runs=8)

    ## Example: creating target files for the action observation knots task
    AO = ActionObservationKnots()
    AO._make_files()









    