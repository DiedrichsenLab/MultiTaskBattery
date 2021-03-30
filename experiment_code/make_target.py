# Create target file for different tasks
# @ Ladan Shahshahani  - Maedbh King March 30 2021
import pandas as pd
import numpy as np

# import experiment_code.constants as consts
import constants as consts



# define classes for target file and run files
class Target():

    def __init__(self, study_name, task_name, hand, trial_dur, iti_dur,
                 run_number, display_trial_feedback = True, task_dur = 30, tr = 1):

        self.study_name             = study_name             # name of the study: 'fmri' or 'behavioral'
        self.task_name              = task_name              # name of the task
        self.task_dur               = task_dur               # duration of the task (default: 30 sec)
        self.hand                   = hand                   # string representing the hand: "right", "left", or "none"
        self.trial_dur              = trial_dur              # duration of trial
        self.iti_dur                = iti_dur                # duration of the inter trial interval
        self.display_trial_feedback = display_trial_feedback # display feedback after trial (default: True)
        self.tr                     = tr                     # the TR of the scanner
        self.run_number             = run_number             # the number of run
        self.target_dict            = {}                     # a dicttionary that will be saved as target file for the task

    def sample_evenly_from_col(self):
        if kwargs.get("random_state"):
            random_state = kwargs["random_state"]
        else: 
            random_state = 2
        num_values = len(dataframe[column].unique())
        group_size = int(np.ceil(num_stim / num_values))

        group_data = dataframe.groupby(column).apply(lambda x: x.sample(group_size, random_state=random_state, replace=False))
        group_data = group_data.sample(num_stim, random_state=random_state, replace=False).reset_index(drop=True).sort_values(column)

        return group_data.reset_index(drop=True)

    def save_target_file(self):

        df = pd.DataFrame(self.target_dict)
        print(df)
        # path to save the target files
        path2task_target = consts.target_dir / self.study_name / self.task_name
        consts.dircheck(path2task_target)

        target_filename = path2task_target / f"{self.task_name}_{self.task_dur}sec_{self.run_number+1:02d}.csv"
        df.to_csv(target_filename)
            
    def make_trials(self):
        self.num_trials = int(self.task_dur/(self.trial_dur + self.iti_dur)) # total number of trials
        self.target_dict['start_time'] = [(self.trial_dur + self.iti_dur)*trial_number for trial_number in range(self.num_trials)]
        self.target_dict['end_time']   = [(trial_number+1)*self.trial_dur + trial_number*self.iti_dur for trial_number in range(self.num_trials)]
        self.target_dict['hand']       = np.tile(self.hand, self.num_trials).T.flatten() 
        self.target_dict['trial_dur']  = [self.trial_dur for trial_number in range(self.num_trials)]
        self.target_dict['iti_dur']    = [self.iti_dur for trial_number in range(self.num_trials)]

class Run():

    def __init__(self):

        self.run_number
        self.task_list
        self.instruct_dur
        self.task_dur
        self.counter_balance

    def make_run_df(self):
        pass

    def save_run_df(self):
        pass

# define classes for each task target file
class FingerSequence(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 3.25,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = True, 
                 task_dur=30, tr = 1, seq_length = 6):

        print(run_number)
        super(FingerSequence, self).__init__(study_name = study_name, task_name = 'finger_sequence', hand = hand, 
                                             trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number, 
                                             display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)
        self.task_name  = "finger_sequence"
        self.seq_length = seq_length

        # the sequences
        self.seq = {}
        ## EXperimental:
        self.seq['complex'] = ['1 3 2 4 3 2', '2 1 3 4 3 1', 
                        '3 2 4 1 4 2', '4 1 2 3 4 1']
        ## Control
        self.seq['simple'] = ['1 1 1 1 1 1', '2 2 2 2 2 2', 
                        '3 3 3 3 3 3', '4 4 4 4 4 4']

    def _add_info(self):
        super().make_trials() # first fill in the common fields

        n_complex = int(self.num_trials/2)
        n_simple  = self.num_trials - n_complex

        # shuffle sequences for complex and simple
        np.random.shuffle(self.seq['complex'])
        np.random.shuffle(self.seq['simple'])

        self.target_dict['trial_type'] = ['None' for trial_number in range(self.num_trials)]

        # fill in fields ?????????????????????????
        ## randomize order of complex and simple conditions across runs
        if self.run_number%2 == 0: # in even runs complex sequences come first
            self.target_dict['condition_type'] = np.concatenate((np.tile('complex', n_complex), np.tile('simple', n_complex)), axis=0)
            self.target_dict['sequence']       = np.concatenate((self.seq['complex'], self.seq['simple']), axis=0).T.flatten()
        else: # in odd runs simple sequences come first
            self.target_dict['condition_type'] = np.concatenate((np.tile('simple', n_simple), np.tile('complex', n_simple)), axis=0)
            self.target_dict['sequence']       = np.concatenate((self.seq['simple'], self.seq['complex']), axis=0).T.flatten()


class SternbergOrder(Target):
    def __init__(self):
        pass

class FlexionExtension(Target):
    def __init__(self):
        pass
class ActionObservationKnots(Target):
    def __init__(self):
        pass


class VisuospatialOrder(Target):
    def __init__(self):
        pass

class VisualSearch(Target):
    def __init__(self):
        pass

class SemanticPrediction(Target):
    def __init__(self):
        pass

class VerbGeneration(Target):
    pass

class ActionObservation(Target):
    pass

class TheoryOfMind(Target):
    pass

class RomanceMovie(Target):
    pass

class Rest(Target):
    pass

class NBack(Target):
    pass

class SocialPrediction(Target):
    pass


# define functions to create target files and run files
def make_target(task_name):
    pass
def make_run(task_list, num_runs):
    pass


TASK_MAP = {
    "visual_search": VisualSearch, # task_num 1
    "theory_of_mind": TheoryOfMind, # task_num 2
    "n_back": NBack, # task_num 3
    "social_prediction": SocialPrediction, # task_num 4
    "semantic_prediction": SemanticPrediction, # task_num 5
    "action_observation": ActionObservation, # task_num 6 
    "finger_sequence": FingerSequence, # task_num 7
    "sternberg_order": SternbergOrder, # task_num 8
    "visuospatial_order": VisuospatialOrder, # task 9
    "flexion_extension": FlexionExtension, # task_num 10
    "verb_generation": VerbGeneration, # task_num 11
    "romance_movie": RomanceMovie, #task_num 12
    "act_obs_knots": ActionObservationKnots, #task_num 13
    "rest": Rest, # task_num?
    }


FS = FingerSequence()
FS._add_info()
FS.save_target_file()



