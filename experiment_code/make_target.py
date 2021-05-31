# Create target file for different tasks
# @ Ladan Shahshahani  - Maedbh King March 30 2021
import pandas as pd
import numpy as np
import os
import random

# import experiment_code.constants as consts
import constants as consts



# define classes for target file and run files
class Target():

    def __init__(self, study_name, task_name, hand, trial_dur, iti_dur,
                 run_number, display_trial_feedback = True, task_dur = 30, tr = 1):

        """
        variables and information shared across all tasks
        """
        self.study_name             = study_name             # name of the study: 'fmri' or 'behavioral'
        self.task_name              = task_name              # name of the task
        self.task_dur               = task_dur               # duration of the task (default: 30 sec)
        self.hand                   = hand                   # string representing the hand: "right", "left", or "none"
        self.trial_dur              = trial_dur              # duration of trial
        self.iti_dur                = iti_dur                # duration of the inter trial interval
        self.display_trial_feedback = display_trial_feedback # display feedback after trial (default: True)
        self.tr                     = tr                     # the TR of the scanner
        self.run_number             = run_number             # the number of run
        self.replace                = False
        self.target_dict            = {}                     # a dicttionary that will be saved as target file for the task

        # file naming stuff
        self.target_filename = f"{self.task_name}_{self.task_dur}sec_{self.run_number:02d}"
        self.target_dir      = consts.target_dir / self.study_name / self.task_name
        consts.dircheck(self.target_dir)

        self.target_filedir = self.target_dir / f"{self.target_filename}.csv"
         
    def make_trials(self):
        """
        making trials (rows) with columns (variables) shared across tasks
        """
        self.num_trials = int(self.task_dur/(self.trial_dur + self.iti_dur)) # total number of trials
        # self.target_dict['start_time'] = [(self.trial_dur + self.iti_dur)*trial_number for trial_number in range(self.num_trials)]
        # self.target_dict['end_time']   = [(trial_number+1)*self.trial_dur + trial_number*self.iti_dur for trial_number in range(self.num_trials)]
        self.target_dict['hand']       = np.tile(self.hand, self.num_trials).T.flatten() 
        self.target_dict['trial_dur']  = [self.trial_dur for trial_number in range(self.num_trials)]
        self.target_dict['iti_dur']    = [self.iti_dur for trial_number in range(self.num_trials)]
        self.target_dict['run_number'] = [self.run_number for trial_number in range(self.num_trials)]
        self.target_dict['display_trial_feedback'] = [self.display_trial_feedback for trial_number in range(self.num_trials)]

    def make_trials_time(self, dataframe):
        """
        adds start_time and end_time columns to the dataframe
        """

        dataframe['start_time'] = [(self.trial_dur + self.iti_dur)*trial_number for trial_number in range(self.num_trials)]
        dataframe['end_time']   = [(trial_number+1)*self.trial_dur + trial_number*self.iti_dur for trial_number in range(self.num_trials)]

        return dataframe
    
    def balance_design(self, dataframe, random_state):
        """
        balancing design
        """
        # groupbyObject.sample returns a random sample of items from each group.
        # dataframe = dataframe.groupby([*self.trials_info], as_index=False).apply(lambda x: x.sample(n=int(self.num_stims), random_state=random_state, replace=self.replace)).reset_index(drop=True)
        dataframe = dataframe.sample(frac = 1, random_state=random_state)
        return dataframe

    def save_target_file(self, dataframe):
        """
        save the target file in the corresponding directory
        """
        dataframe.to_csv(self.target_filedir)

class Run():

    def __init__(self):

        self.run_number
        self.task_list
        self.instruct_dur
        self.task_dur
        self.num_runs 
        self.tile_runs
        self.counter_balance

    def make_run_df(self):
        """
        makes run dataframe
        """
        pass

    def save_run_df(self):
        """
        saves run dataframe
        """
        pass

    def test_counterbalance(self):
        """
        tests counterbalancing of tasks across runs
        """
        pass

# define classes for each task target file
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
        self.seq['complex'] = ['1 3 2 4 3 2', '2 1 3 4 3 1', 
                        '3 2 4 1 4 2', '4 1 2 3 4 1']
        ## Control
        self.seq['simple'] = ['1 1 1 1 1 1', '2 2 2 2 2 2', 
                        '3 3 3 3 3 3', '4 4 4 4 4 4']

    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        n_complex = int(self.num_trials/2)
        n_simple  = self.num_trials - n_complex

        # shuffle sequences for complex and simple
        np.random.shuffle(self.seq['complex'])
        np.random.shuffle(self.seq['simple'])

        self.target_dict['trial_type'] = ['None' for trial_number in range(self.num_trials)]

        # fill in fields ?????????????????????????
        # ## randomize order of complex and simple conditions across runs
        # if self.run_number%2 == 0: # in even runs complex sequences come first
        #     self.target_dict['condition_name'] = np.concatenate((np.tile('complex', n_complex), np.tile('simple', n_complex)), axis=0)
        #     self.target_dict['sequence']       = np.concatenate((self.seq['complex'], self.seq['simple']), axis=0).T.flatten()
        # else: # in odd runs simple sequences come first
        #     self.target_dict['condition_name'] = np.concatenate((np.tile('simple', n_simple), np.tile('complex', n_simple)), axis=0)
        #     self.target_dict['sequence']       = np.concatenate((self.seq['simple'], self.seq['complex']), axis=0).T.flatten()

        self.target_dict['condition_name'] = np.concatenate((np.tile('complex', n_complex), np.tile('simple', n_complex)), axis=0)
        self.target_dict['sequence']       = np.concatenate((self.seq['complex'], self.seq['simple']), axis=0).T.flatten()
        self.target_dict['feedback_type']  = [self.feedback_type for trial_number in range(self.num_trials)]

        dataframe = pd.DataFrame(self.target_dict) # convert to dataframe

        # randomly shuffle the rows of the dataframe
        dataframe = dataframe.sample(frac = 1, random_state=random_state, axis = 0).reset_index(drop=True)

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
    
    def _get_prob_stims(self):
        # determine the prob stim for each trial based on trial type
        self.prob_stim = []
        for trial in range(self.num_trials):
            # get the trial_type for the current trial
            current_tt = self.target_dict['trial_type'][trial]

            # get the current stims
            current_stim = self.target_dict['stim'][trial]
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
        self.target_dict['trial_type'] = (int(self.num_trials/len(self.trials_info['trial_type'])))*self.trials_info['trial_type']

        self.target_dict['delay_dur'] = [self.delay_dur for i in range(self.num_trials)]
        self.target_dict['digit_dur'] = [self.digit_dur for i in range(self.num_trials)]
        self.target_dict['prob_dur']  = [self.prob_dur for i in range(self.num_trials)]

        # generate random numbers betweem 1 and 9 
        rand_nums = [np.random.choice(range(1, 10), size = 6, replace = False) for i in range(self.num_trials)]
        ## convert the random numbers to str and concatenate them
        stim_str = []
        for nums in rand_nums:
            rand_str = ""
            for x in nums: rand_str += str(x) + " "
            stim_str.append(rand_str)

        self.target_dict['stim'] = stim_str

        # get the prob stimulus of trials
        self._get_prob_stims()

        self.target_dict['prob_stim'] = self.prob_stim
        dataframe = pd.DataFrame(self.target_dict) # convert to dataframe
        # dataframe['trial_type'] = dataframe['trial_type'].sort_values().reset_index(drop=True)

        # randomly shuffle rows of the dataframe
        dataframe = dataframe.sample(frac = 1, random_state=random_state, axis = 0).reset_index(drop=True)

        return dataframe

    def _make_files(self):
        """
        makes target file and (if exists) related task info and  saves them
        """

        # save target file
        self.df = self._add_task_info(random_state=self.run_number)
        self.df = self.make_trials_time(self.df)
        self.save_target_file(self.df)

class FlexionExtension(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 14,
                 iti_dur = 1, run_number = 1, display_trial_feedback = False, 
                 task_dur = 30, stim_dur = 1, tr = 1):

        super(FlexionExtension, self).__init__(study_name = study_name, task_name = 'flexion_extension', hand = hand, 
                                               trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number, 
                                               display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)

        self.trials_info = {"condition_name":["flexion extention"], "trial_type":[None]}
        self.stim_dur = stim_dur # time while either flexion or extension is remaining on the screen

    def _add_task_info(self, random_state):
        super().make_trials() # first fill in the common fields

        self.target_dict['stim']       = ["flexion extension" for i in range(self.num_trials)]
        self.target_dict['stim_dur']   = [self.stim_dur for i in range(self.num_trials)]
        self.target_dict['trial_type'] = ['None' for i in range(self.num_trials)] # there are no true of false responses
        self.target_dict['hand']       = ['None' for i in range(self.num_trials)] # hand is not used

        dataframe = pd.DataFrame(self.target_dict) # convert to dataframe

        # randomly shuffle the rows of the dataframe
        dataframe = dataframe.sample(frac = 1, random_state=random_state, axis = 0).reset_index(drop=True)

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
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 3.25,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = True, 
                 task_dur=30, tr = 1, seq_length = 6):

        super(ActionObservationKnots, self).__init__(study_name = study_name, task_name = 'act_obs_knots', hand = hand, 
                                             trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number, 
                                             display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)

class VisuospatialOrder(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', trial_dur = 6,
                 iti_dur = 0.5, run_number = 1, display_trial_feedback = True, 
                 task_dur=30, tr = 1, dot_dur = 0.75, delay_dur = 0.5, 
                 prob_dur = 1, load = 6, min_distance = 1, width = 8):
        super(VisuospatialOrder, self).__init__(study_name = study_name, task_name = 'visuospatial_order', hand = hand, 
                                                trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number, 
                                                display_trial_feedback = display_trial_feedback, task_dur = task_dur, tr = tr)

        
        self.trials_info = {'condition_name':[], "trial_type":[True, False]}
        self.prob_dur     = prob_dur     # length of time the prob remains on the screen (also time length for response to be made)
        self.digit_dur    = dot_dur      # length of time each digit remains on the screen
        self.delay_dur    = delay_dur    # duration of the delay
        self.load         = load         # memory load
        self.width        = width        # width of the enclosing square (or can be circle)
        self.min_distance = min_distance # minimum distance between dots

    def _add_task_info(self):
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
        self.target_dict['trial_type'] = np.random.choice([True, False], size = self.num_trials, replace=True)
        # ------------------------------------------------------------------

        self.target_dict['delay_dur'] = [self.delay_dur for i in range(self.num_trials)]
        self.target_dict['dot_dur']   = [self.digit_dur for i in range(self.num_trials)]
        self.target_dict['prob_dur']  = [self.prob_dur for i in range(self.num_trials)]
        self.target_dict['width']     = [self.width for i in range(self.num_trials)]

        # creating empty lists for coordinates of dots
        self.target_dict['xyz_stim'] = [] # for the stimulus
        self.target_dict['xyz_prob'] = [] # for the probe

        # loop over trials and create coordinates of stim and prob
        for t in range(self.num_trials):
            # ------------------- Generate random coordinates -------------------------
            # generate random points
            x = np.random.uniform(-self.width/2, self.width/2) 
            y = np.random.uniform(-self.width/2, self.width/2)

            # checks the distance between all the points to make sure they are at a minimum distance
            counter  = 2 # counter for the point
            dot_xys = []
            dot_xys.append([x, y])
            while counter < self.load + 1:
                # start generating the other points
                ## generate another random point 
                next_point = False
                xi = np.random.uniform(-self.width/2, self.width/2) 
                yi = np.random.uniform(-self.width/2, self.width/2)

                # check the distance between this point and all the other points already in the list dot_xys
                for point in dot_xys:
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
                    dot_xys.append([xi, yi])

            self.target_dict['xyz_stim'].append(dot_xys) 

            # randomly pick two of the dots for probe based on the trial type
            # get the trial_type for the current trial
            current_tt = self.target_dict['trial_type'][t]  

            # pick two dots
            rand_probs = np.random.choice(len(dot_xys), size = 2, replace = False) 

            if ~ current_tt: # False trial
                # the trial is false so two wrong digits with wrong order can be generated
                # sort the indices in descending order to make sure that their order is flipped
                probs_idx = np.sort(rand_probs)[::-1]
            else: # True trial
                # sort the indices in ascending order to make sure that their order is conserved
                probs_idx = np.sort(rand_probs)
            
            probs_xys  = [dot_xys[i] for i in probs_idx]   

            self.target_dict['xys_prob'].append(probs_xys) 

class VisualSearch(Target):
    def __init__(self, study_name = 'behavioral', hand = 'right', 
                 trial_dur = 2, iti_dur = 0.5, run_number = 1, display_trial_feedback = True, 
                 task_dur = 30, tr = 1, replace = False):

        super(VisualSearch, self).__init__(study_name = study_name, task_name = 'visual_search', hand = hand, 
                                           trial_dur = trial_dur, iti_dur = iti_dur, run_number = run_number, 
                                           display_trial_feedback = display_trial_feedback, task_dur = task_dur,
                                           tr = tr)

        self.feedback_type  = 'rt'
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
        
        self.target_dict['stim']            = (int(self.num_trials/len(conds)))*conds
        self.target_dict['condition_name']  = (int(self.num_trials/len(conds_names)))*conds_names
        self.target_dict['trial_type']      = (int(self.num_trials/len(self.trials_info['trial_type'])))*self.trials_info['trial_type']
        self.target_dict['feedback_type']   = [self.feedback_type for trial_number in range(self.num_trials)]
        

        dataframe = pd.DataFrame(self.target_dict) # convert to dataframe
        # dataframe['trial_type'] = dataframe['trial_type'].sort_values().reset_index(drop=True)
        dataframe['stim']       = dataframe['stim'].astype(int) # convert stim to int 

        # randomly shuffle rows of the dataframe
        dataframe = dataframe.sample(frac = 1, random_state=random_state, axis = 0).reset_index(drop=True)

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
        visual_display_name = 'display_pos' + str_part[2]+".csv" 
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


VS = VisualSearch()
VS._make_files()

FS = FingerSequence()
FS._make_files()

SO = SternbergOrder()
SO._make_files()

FE = FlexionExtension()
FE._make_files()


