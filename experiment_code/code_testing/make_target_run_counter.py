from pathlib import Path

import os
import re
import pandas as pd
import numpy as np
import random
from math import ceil
import copy
import cv2
import glob
import shutil
import json
from itertools import count
from collections import OrderedDict

from experiment_code.fmri.constants import Defaults

import warnings
warnings.filterwarnings('ignore')

class MakeFiles:
    """
    This class makes run and target files
        Args:
            block_names (list of str): options are 'visual_search', 'n_back', 'social_prediction', 'semantic_prediction', 'action_observation'
            run_name_prefix (str): prefix of run name
            tile_run (int): determines number of block repeats within a run
            instruct_dur (int): length of instruct for block_names (sec)
            block_dur_secs (int): length of block_name (sec)
            rest_dur_secs (int): length of rest (sec), 0 if no rest
            num_runs (int): number of runs
            counterbalance_runs (bool): counterbalance block order across runs
    """

    def __init__(self, **kwargs):
        f = open(file=os.path.join(Defaults.CONFIG_DIR, 'run_config.json'))
        config = json.load(f)
        self.config = copy.deepcopy(config)
        self.config.update(**kwargs)

    def _create_run_dataframe(self, target_files):
        for iter, target_file in enumerate(target_files):

            # load target file
            dataframe = pd.read_csv(target_file)

            start_time = dataframe.iloc[0]['start_time'] + self.cum_time
            end_time = dataframe.iloc[-1]['start_time'] + dataframe.iloc[-1]['trial_dur'] + self.config['instruct_dur'] + self.cum_time

            target_file_name = Path(target_file).name
            num_sec = re.findall(r'\d+(?=sec)', target_file)[0]
            target_num = re.findall(r'\d+(?=.tsv)', target_file)[0]
            num_trials = len(dataframe)

            data = {'block_name': self.block_name, 'block_iter': iter+1, 'block_num': self.block_num+1, # 'block_iter': iter+1
                    'num_trials': num_trials, 'target_num': target_num, 'num_sec': num_sec,
                    'target_file': target_file_name, 'start_time': start_time, 'end_time': end_time,
                    'instruct_dur': self.config['instruct_dur'], 'display_trial_feedback': self.display_trial_feedback,
                    'replace_stimuli': self.replace_stimuli, 'feedback_type': self.feedback_type, 'target_score': self.target_score}

            self.all_data.append(data)
            self.cum_time = end_time

    def _save_run_file(self, dataframe, run_name):
        # save out to file
        dataframe.to_csv(os.path.join(Defaults.RUN_DIR, run_name), index=False, header=True)

    def _add_rest(self):
        run_name_prefix = self.config['run_name_prefix']
        run_files = sorted(glob.glob(os.path.join(Defaults.RUN_DIR, f'*{run_name_prefix}*.tsv')))

        # make target file
        BlockClass = TASK_MAP['rest']
        config = self._load_config(fpath=os.path.join(Defaults.CONFIG_DIR, f'rest_config.json'))
        block = BlockClass(target_config=config)
        self.target_name = block.make_targetfile()

        for run_file in run_files:
            dataframe = pd.read_csv(run_file)

            dataframe = self._add_rest_rows(dataframe)

            dataframe.to_csv(run_file, index = False, header = True)

    def _counterbalance_runs(self):
        while self._test_counterbalance() > 0:
            print('not balanced ...')
            self._create_run()

        print('these runs are perfectly balanced')

    def _check_task_run(self):
        # check if task exists in dict
        exists_in_dict = [True for key in self.target_dict.keys() if self.block_name==key]
        if not exists_in_dict:
            self.target_dict.update({self.block_name: self.fpaths})

        # create run dataframe
        random.seed(self.block_num+1)
        target_files_sample = [self.target_dict[self.block_name].pop(random.randrange(len(self.target_dict[self.block_name]))) for _ in np.arange(self.config['tile_run'])]

        return target_files_sample

    def _insert_row(self, row_number, dataframe, row_value):
        # Slice the upper half of the dataframe
        df1 = dataframe[0:row_number]

        # Store the result of lower half of the dataframe
        df2 = dataframe[row_number:]

        # Insert the row in the upper half dataframe
        df1.loc[row_number]=row_value

        # Concat the two dataframes
        df_result = pd.concat([df1, df2])

        # Reassign the index labels
        df_result.index = [*range(df_result.shape[0])]

        # Return the updated dataframe
        return df_result

    def _correct_start_end_times(self, dataframe):

        timestamps = (np.cumsum(dataframe['num_sec'] + dataframe['instruct_dur'])).to_list()

        dataframe['end_time'] = timestamps

        timestamps.insert(0, 0)
        dataframe['start_time'] = timestamps[:-1]

        return dataframe

    def _add_rest_rows(self, dataframe):
        self.num_rest = (len(self.config['block_names']) * self.config['tile_run']) - 1

        trials_before_rest = np.tile(np.round((len(dataframe) + self.num_rest) /(self.num_rest)), self.num_rest)
        rest = np.cumsum(trials_before_rest).astype(int) - 1

        # row values
        row_dict = {'block_name': 'rest', 'block_iter': np.float('NaN') , 'block_num': len(self.config['block_names']) + 1,
                     'num_trials': 1, 'target_num': np.float('NaN') , 'num_sec': self.config['rest_dur_secs'],
                    'target_file': self.target_name, 'start_time': np.float('NaN') , 'end_time': np.float('NaN') ,
                    'instruct_dur': 0, 'display_trial_feedback': np.float('NaN') ,
                    'replace_stimuli': np.float('NaN') , 'feedback_type': np.float('NaN') , 'target_score': np.float('NaN') }

        rest_blocks = np.arange(1, self.num_rest+1)

        # Let's create a row which we want to insert
        for idx, row_number in enumerate(rest):
            # row_value = np.tile('rest', len(dataframe.columns))
            row_dict.update({'block_iter': rest_blocks[idx]})
            row_value = list(row_dict.values())
            if row_number > dataframe.index.max()+1:
                print("Invalid row_number")
            else:
                dataframe = self._insert_row(row_number, dataframe, row_value)

        # update start and end times
        dataframe = self._correct_start_end_times(dataframe)

        return dataframe

    def _load_config(self, fpath):
        """ loads JSON file as dict
            Args:
                fpath (str): full path to .json file
            Returns
                loads JSON as dict
        """
        f = open(fpath)

        # returns JSON object as a dict
        return json.load(f)

    def _save_target_files(self, df_target):
        """ saves out target files
            Args:
                df_target (pandas dataframe)
            Returns:
                modified pandas dataframes `df_target`
        """
        # # shuffle and set a seed (to ensure reproducibility)
        # df_target = df_target.sample(n=len(df_target), random_state=self.random_state, replace=False).reset_index(drop=True)

        start_time = np.round(np.arange(0, self.num_trials*(self.config['trial_dur']+self.config['iti_dur']), self.config['trial_dur']+self.config['iti_dur']), 1)
        data = {"trial_dur":self.config['trial_dur'], "iti_dur":self.config['iti_dur'], "start_time":start_time, "hand": self.config['hand']}

        df_target = pd.concat([df_target, pd.DataFrame.from_records(data)], axis=1, ignore_index=False, sort=False)

        # get targetfile name
        tf_name = f"{self.config['block_name']}_{self.config['block_dur_secs']}sec" # was {self.num_trials}trials
        tf_name = self._get_target_file_name(tf_name)

        # save out dataframe to a csv file in the target directory (TARGET_DIR)
        df_target.to_csv(os.path.join(self.TARGET_DIR, tf_name), index=False, header=True)

        print(f'saving out {tf_name}')

    def _get_target_file_name(self, targetfile_name):
        # figure out naming convention for target files
        target_num = []

        if not os.path.exists(self.TARGET_DIR):
            os.makedirs(self.TARGET_DIR)

        for f in os.listdir(self.TARGET_DIR):
            if re.search(targetfile_name, f):
                regex = r"_(\d+).tsv"
                target_num.append(int(re.findall(regex, f)[0]))

        if target_num==[]:
            outfile_name = f"{targetfile_name}_01.tsv" # first target file
        else:
            num = np.max(target_num)+1
            outfile_name = f"{targetfile_name}_{num:02d}.tsv" # second or more

        return outfile_name

    def _sample_evenly_from_col(self, dataframe, num_stim, column='condition_name', **kwargs):
        if kwargs.get('random_state'):
            random_state = kwargs['random_state']
        else:
            random_state = 2

        if kwargs.get('replace'):
            replace = kwargs['replace']
        else:
            replace = False
        num_values = len(dataframe[column].unique())
        group_size = int(np.ceil(num_stim / num_values))
        group_data = dataframe.groupby(column).apply(lambda x: x.sample(group_size, random_state=random_state, replace=replace))
        group_data = group_data.sample(num_stim, random_state=random_state, replace=replace).reset_index(drop=True).sort_values(column)
        return group_data.reset_index(drop=True)

    def _correct_block_iter(self, dataframe):
        dataframe['block_iter'] = dataframe.groupby('block_name').cumcount() + 1

        return dataframe

    def _create_run(self):
        # delete any run files that exist in the folder
        # files = glob.glob(os.path.join(Defaults.RUN_DIR, '*run*.tsv'))
        # for f in files:
        #     os.remove(f)

        # create run files
        self.target_dict = {}
        for run in np.arange(self.config['num_runs']):
            self.cum_time = 0.0
            self.all_data = []

            for self.block_num, self.block_name in enumerate(self.config['block_names']):

                # get target files for `block_name`
                self.TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.block_name)
                self.fpaths = sorted(glob.glob(os.path.join(self.TARGET_DIR, f'*{self.block_name}*.tsv')))

                # sample tasks
                target_files_sample = self._check_task_run()

                # get tf info
                df = pd.read_csv(os.path.join(self.TARGET_DIR, target_files_sample[0]))
                self.display_trial_feedback = np.unique(df['display_trial_feedback'])[0]
                self.replace_stimuli = np.unique(df['replace_stimuli'])[0]
                self.feedback_type = np.unique(df['feedback_type'])[0]
                self.target_score = np.unique(df['target_score'])[0]

                # create run dataframe
                self._create_run_dataframe(target_files=target_files_sample)

            # shuffle order of tasks within run
            df_run = pd.DataFrame.from_dict(self.all_data)
            df_run = df_run.sample(n=len(df_run), replace=False)

            # correct `block_iter`, `start_time`, `run_time`
            df_run = self._correct_block_iter(dataframe=df_run)
            df_run['start_time'] = sorted(df_run['start_time'])
            df_run['end_time'] = sorted(df_run['end_time'])

            # save run file
            run_name = self.config['run_name_prefix'] + '_' +  f'{run+1:02d}' + '.tsv'
            self._save_run_file(dataframe=df_run, run_name=run_name)
            # print(f'saving out {run_name}')

    def _test_counterbalance(self):
        filenames = sorted(glob.glob(os.path.join(Defaults.RUN_DIR, '*run_*')))

        dataframe_all = pd.DataFrame()
        for i, file in enumerate(filenames):
            dataframe = pd.read_csv(file)
            dataframe['run'] = i + 1
            dataframe['block_num_unique'] = np.arange(len(dataframe)) + 1
            dataframe_all = pd.concat([dataframe_all, dataframe])

        # create new column
        dataframe_all['block_name_unique'] = dataframe_all['block_name'] + '_' + dataframe_all['block_iter'].astype(str)

        task = np.array(list(map({}.setdefault, dataframe_all['block_name_unique'], count()))) + 1
        last_task = list(task[0:-1])
        last_task.insert(0,0)
        last_task = np.array(last_task)
        last_task[dataframe_all['block_num_unique']==1] = 0

        dataframe_all['last_task'] = last_task
        dataframe_all['task'] = task
        dataframe_all['task_num'] = task

        # get pivot table
        f = pd.pivot_table(dataframe_all, index=['task'], columns=['last_task'], values=['task_num'], aggfunc=len)

        return sum([sum(f['task_num'][col]>5) for col in f['task_num'].columns])

    def check_videos(self):
        for block_name in ['action_observation', 'social_prediction']:

            TARGET_DIR = os.path.join(Defaults.TARGET_DIR, block_name)

            files = glob.glob(os.path.join(Defaults.TARGET_DIR, block_name, '*.tsv'))

            # loop over files
            video_count = []
            for file in files:
                dataframe = pd.read_csv(file)

                # loop over videos and check that they exist
                videos = dataframe['stim']
                for stim in videos:
                    video_fpath = os.path.join(Defaults.STIM_DIR, block_name, "modified_clips", stim)
                    if not os.path.exists(video_fpath):
                        video_count.append(stim)
                        print(f'{block_name}: {stim} is missing from videos')

            if not video_count:
                print(f'there are no videos missing for {block_name}')

    def make_targetfiles(self, **kwargs):
        for self.block_name in self.config['block_names']:

            TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.block_name)

            # delete any target files that exist in the folder
            files = glob.glob(os.path.join(Defaults.TARGET_DIR, self.block_name, '*.tsv'))
            for f in files:
                os.remove(f)

            # make target files
            BlockClass = TASK_MAP[self.block_name]
            config = self._load_config(fpath=os.path.join(Defaults.CONFIG_DIR, f'{self.block_name}_config.json'))
            block = BlockClass(target_config=config, **kwargs)
            block.make_targetfile()

    def make_runfiles(self, **kwargs):

        # make run files
        self._create_run()

        # OPTION TO COUNTERBALANCE RUNS
        if self.config['counterbalance_runs']:
            self._counterbalance_runs()

        # OPTION TO ADD REST
        if self.config['rest_dur_secs']>0:
            self._add_rest()

        # check if videos for action observation and social prediction exist
        self.check_videos()

    def make_all(self, **kwargs):
        # create target files
        self.make_targetfiles(**kwargs)

        # create run files
        self.make_runfiles(**kwargs)

class VisualSearch(MakeFiles):
    """
        This class makes target files for Visual Search using parameters from config file
        Args:
            target_config (dict): dictionary loaded from `visual_search_config.json`
        Kwargs:
            block_name (str): 'visual_search'
            orientations (int): orientations of target/distractor stims
            balance_blocks (dict): keys are 'condition_name', 'trial_type'
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for block_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """

    def __init__(self, target_config, **kwargs):
        super().__init__()
        self.config.update(target_config)
        self.config.update(**kwargs)

    def _get_block_info(self):
        # num of blocks (i.e. target files) to make
        self.num_blocks = self.config['num_runs'] * self.config['tile_run']

        # get overall number of trials
        self.num_trials = int(self.config['block_dur_secs'] / (self.config['trial_dur'] + self.config['iti_dur']))

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.config['balance_blocks'].values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int

    def _create_columns(self):

        def _get_condition(x):
            for key in self.config['balance_blocks']['condition_name'].keys():
                cond = self.config['balance_blocks']['condition_name'][key]
                if x==cond:
                    value = key
            return value

        dataframe = pd.DataFrame()
        # make `condition_name` column
        conds = [self.config['balance_blocks']['condition_name'][key] for key in self.config['balance_blocks']['condition_name'].keys()]
        dataframe['stim'] = self.num_trials*conds
        dataframe['condition_name'] = dataframe['stim'].apply(lambda x: _get_condition(x))
        dataframe['stim'] = dataframe['stim'].astype(int)

        # make `trial_type` column
        dataframe['trial_type'] = self.num_trials*self.config['balance_blocks']['trial_type']
        dataframe['trial_type'] = dataframe['trial_type'].sort_values().reset_index(drop=True)

        dataframe['display_trial_feedback'] = self.config['display_trial_feedback']
        dataframe['replace_stimuli'] = self.config['replace']
        dataframe['feedback_type'] = self.config['feedback_type']
        dataframe['target_score'] = self.config['target_score']

        return dataframe

    def _balance_design(self, dataframe):

        # this assumes that there is a `condition_name` key in all tasks (which there should be)
        # dataframe = dataframe.groupby([*self.config['balance_blocks']], as_index=False).apply(lambda x: self._sample_evenly_from_col(x, num_stim=self.num_stims, column='condition_name', random_state=self.random_state, replace=self.config['replace'])).reset_index(drop=True)

        dataframe =  dataframe.groupby([*self.config['balance_blocks']], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.config['replace'])).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        num_stims = int(self.num_trials / len(self.config['balance_blocks']['condition_name']))
        dataframe = dataframe.groupby('condition_name', as_index=False).apply(lambda x: x.sample(n=num_stims, random_state=self.random_state, replace=False)).reset_index(drop=True)

        # shuffle the order of the trials
        dataframe = dataframe.apply(lambda x: x.sample(n=self.num_trials, random_state=self.random_state, replace=False)).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        return dataframe

    def _save_visual_display(self, dataframe):
        # add visual display cols
        display_pos, orientations_correct = zip(*[self._make_search_display(cond, self.config['orientations'], trial_type) for (cond, trial_type) in zip(dataframe["stim"], dataframe["trial_type"])])

        data_dicts = []
        for trial_idx, trial_conditions in enumerate(display_pos):
            for condition, point in trial_conditions.items():
                data_dicts.append({'trial': trial_idx, 'stim': condition, 'xpos': point[0], 'ypos': point[1], 'orientation': orientations_correct[trial_idx][condition]})

        # save out to dataframe
        df_display = pd.DataFrame.from_records(data_dicts)

        # save out visual display
        visual_display_name = self._get_visual_display_name()
        df_display.to_csv(os.path.join(self.TARGET_DIR, visual_display_name))

    def _get_visual_display_name(self):
        block_name = self.config['block_name']
        block_dur_secs = self.config['block_dur_secs']
        tf_name = f"{block_name}_{block_dur_secs}sec"
        tf_name = self._get_target_file_name(tf_name)

        str_part = tf_name.partition(self.config['block_name'])
        visual_display_name = 'display_pos' + str_part[2]

        return visual_display_name

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

    def make_targetfile(self):
        """
        makes target file(s) for action observation
        """
        # get info about block
        self._get_block_info()

        seeds = np.arange(self.num_blocks)+1

        for self.block in np.arange(self.num_blocks):

            # randomly sample so that conditions (2Back- and 2Back+) are equally represented
            self.random_state = seeds[self.block]

            # create the dataframe
            df_target = self._create_columns()

            # balance the dataframe
            df_target = self._balance_design(dataframe=df_target)

            self.TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.config['block_name'])

            # save visual display dataframe
            self._save_visual_display(dataframe=df_target)

            # save target file
            self._save_target_files(df_target)

class NBack(MakeFiles):
    """
        This class makes target files for NBack using parameters from config file
        Args:
            target_config (dict): dictionary loaded from `n_back_config.json`
        Kwargs:
            block_name (str): 'n_back'
            n_back (int): default is 2
            balance_blocks (dict): keys are 'condition_name'
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for block_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """

    def __init__(self, target_config, **kwargs):
        super().__init__()
        self.config.update(target_config)
        self.config.update(**kwargs)

    def _get_block_info(self, **kwargs):
        # num of blocks (i.e. target files) to make
        self.num_blocks = self.config['num_runs'] * self.config['tile_run']

        # get overall number of trials
        self.num_trials = int(self.config['block_dur_secs'] / (self.config['trial_dur'] + self.config['iti_dur']))

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.config['balance_blocks'].values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int

    def _create_columns(self):

        def _get_condition(x):
            for key in self.config['balance_blocks']['condition_name'].keys():
                cond = self.config['balance_blocks']['condition_name'][key]
                if x==cond:
                    value = key
            return value

        # make trial_type column
        dataframe = pd.DataFrame()
        dataframe['trial_type'] = self.num_stims*(True, False)
        dataframe = dataframe.sample(n=self.num_trials, random_state=self.random_state, replace=False).reset_index(drop=True)
        dataframe['trial_type'][:self.config['n_back']] = False # first n+cond_type trials (depending on cond_type) have to be False

        # make `n_back` and `condition_name` cols
        conds = [self.config['balance_blocks']['condition_name'][key] for key in self.config['balance_blocks']['condition_name'].keys()]
        dataframe['n_back'] = np.where(dataframe["trial_type"]==False, conds[0], conds[1])
        dataframe['condition_name'] = dataframe['n_back'].apply(lambda x: _get_condition(x))

        dataframe['display_trial_feedback'] = self.config['display_trial_feedback']
        dataframe['replace_stimuli'] = self.config['replace']
        dataframe['feedback_type'] = self.config['feedback_type']
        dataframe['target_score'] = self.config['target_score']

        return dataframe

    def _balance_design(self, dataframe):
        # load in stimuli
        stim_files = [f for f in os.listdir(str(Defaults.STIM_DIR / self.config['block_name'])) if f.endswith('g')]

        # first two images are always random (and false)
        # all other images are either match or not a match
        random.seed(self.random_state)
        stim_list = random.sample(stim_files, k=self.config['n_back'])
        for t in dataframe['trial_type'][self.config['n_back']:]: # loop over n+self.n_back
            match_img = stim_list[-self.config['n_back']]
            no_match_imgs = [stim for stim in stim_files if stim != match_img] # was match_img[0]
            if t == False: # not a match
                random.seed(self.random_state)
                stim_list.append(random.sample(no_match_imgs, k=self.config['n_back']-1))
            else:      # match
                stim_list.append(match_img)

        dataframe["stim"] = [''.join(x) for x in stim_list]

        return dataframe

    def make_targetfile(self):
        """
        makes target file(s) for action observation

        """
        # get info about block
        self._get_block_info()

        seeds = np.arange(self.num_blocks)+1

        for self.block in np.arange(self.num_blocks):

            # randomly sample so that conditions (2Back- and 2Back+) are equally represented
            self.random_state = seeds[self.block]

            # create the dataframe
            df_target = self._create_columns()

            # balance the dataframe
            df_target = self._balance_design(dataframe=df_target)

            self.TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.config['block_name'])
            self._save_target_files(df_target)

class SocialPrediction(MakeFiles):
    """
        This class makes target files for Social Prediction using parameters from config file
        Args:
            target_config (dict): dictionary loaded from `social_prediction_config.json`
        Kwargs:
            block_name (str): 'social_prediction'
            dataset_name (str): 'homevideos' is the default
            logging_file (str): csv file containing info about stimuli
            video_name (list of str): name of video(s) to include
            resized (bool): resize frames of video
            balance_blocks (dict): keys are 'actors', 'condition_name', 'label'
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for block_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """

    def __init__(self, target_config, **kwargs):
        super().__init__()
        self.config.update(target_config)
        self.config.update(**kwargs)

    def _filter_dataframe(self, dataframe):
        # remove all filenames where any of the videos have not been extracted
        stims_to_remove = dataframe.query('extracted==False')["video_name"].to_list()
        df_filtered = dataframe[~dataframe["video_name"].isin(stims_to_remove)]

        # query rows with relevant videos and relevant labels
        label = self.config['balance_blocks']['label']
        actors = self.config['balance_blocks']['actors']
        video_name = self.config['video_name']
        df_filtered = df_filtered.query(f'condition_name=={video_name} and label=={label} and actors=={actors}')

        return df_filtered

    def _create_new_columns(self, dataframe):
        # make new `stim`
        if self.config['resized']:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '_resized' + '.mp4'
        else:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '.mp4'

        # set `condition name`
        dataframe['condition_name'] = dataframe['condition_name'].apply(lambda x: self.config['balance_blocks']['condition_name'][x])

        # assign dataset name
        dataframe['dataset'] = self.config['dataset_name']

        # assign trial type (only works currently for two labels)
        labels = self.config['balance_blocks']['label']
        if len(labels)==2:
            dataframe['trial_type'] = dataframe['label'].apply(lambda x: True if x==labels[0] else False)
        else:
            print(f'there are an incorrect number of labels, there should be two')

        dataframe['display_trial_feedback'] = self.config['display_trial_feedback']
        dataframe['replace_stimuli'] = self.config['replace']
        dataframe['feedback_type'] = self.config['feedback_type']
        dataframe['target_score'] = self.config['target_score']

        return dataframe

    def _balance_design(self, dataframe):

        # this assumes that there is a `condition_name` key in all tasks (which there should be)
        dataframe = dataframe.groupby([*self.config['balance_blocks']], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.config['replace'])).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        num_stims = int(self.num_trials / len(self.config['balance_blocks']['condition_name']))
        dataframe = dataframe.groupby('condition_name', as_index=False).apply(lambda x: x.sample(n=num_stims, random_state=self.random_state, replace=False)).reset_index(drop=True)

        # shuffle the order of the trials
        dataframe = dataframe.apply(lambda x: x.sample(n=self.num_trials, random_state=self.random_state, replace=False)).reset_index(drop=True)

        return dataframe

    def _get_block_info(self):
        # num of blocks (i.e. target files) to make
        self.num_blocks = self.config['num_runs'] * self.config['tile_run']

        # get overall number of trials
        self.num_trials = int(self.config['block_dur_secs'] / (self.config['trial_dur'] + self.config['iti_dur']))

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.config['balance_blocks'].values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int

    def make_targetfile(self):
        """
        makes target file(s) for action observation

        """
        # get info about block
        self._get_block_info()

        # return logging file
        fpath = os.path.join(Defaults.STIM_DIR, self.config['block_name'], self.config['logging_file'])

        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        # filter dataframe
        df_filtered = self._filter_dataframe(dataframe=df)

        # create new columns (`trial_type` etc)
        df_filtered = self._create_new_columns(dataframe=df_filtered)

        seeds = np.arange(self.num_blocks)+1

        # for self.block, self.key in enumerate(self.block_design):
        for self.block in np.arange(self.num_blocks):

            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # balance the dataframe by `condition_name` and `player_num`
            df_target = self._balance_design(dataframe=df_filtered)

            # remove `df_target` rows from the main dataframe so that we're always sampling from unique rows
            if self.config['replace']==False:
                df_filtered = df_filtered.merge(df_target, how='left', indicator=True)
                df_filtered = df_filtered[df_filtered['_merge'] == 'left_only'].drop('_merge', axis=1)

            self.TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.config['block_name'])
            self._save_target_files(df_target)

class SemanticPrediction(MakeFiles):
    """
        This class makes target files for Semantic Prediction using parameters from config file
        Args:
            target_config (dict): dictionary loaded from `semantic_prediction_config.json`

        Kwargs:
            block_name (str): 'semantic_prediction'
            logging_file (str): csv file containing info about stimuli
            stem_word_dur (int): length of stem word (sec)
            last_word_dur (int): length of last word (sec)
            frac (int): proportion of meaningless trials. default is .3.
            balance_blocks (dict): keys are 'CoRT_descript', 'condition_name'
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for block_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """

    def __init__(self, target_config, **kwargs):
        super().__init__()
        self.config.update(target_config)
        self.config.update(**kwargs)

    def _filter_dataframe(self, dataframe):
        # conds = [self.balance_blocks['condition_name'][key] for key in self.balance_blocks['condition_name'].keys()]
        conds = list(self.config['balance_blocks']['condition_name'].keys())
        # balance_blocks_cort = self.config['balance_blocks']["CoRT_descript"]
        dataframe = dataframe.query(f'cloze_descript=={conds}')
        # dataframe = dataframe.query(f'CoRT_descript=={balance_blocks_cort} and cloze_descript=={conds}')

        # strip erroneous characters from sentences
        dataframe['stim'] = dataframe['full_sentence'].str.replace('|', ' ')

        return dataframe

    def _create_new_columns(self, dataframe):
        # add condition column
        dataframe['condition_name'] = dataframe['cloze_descript'].apply(lambda x: self.config['balance_blocks']['condition_name'][x])
        dataframe['stem_word_dur'] = self.config['stem_word_dur']
        dataframe['last_word_dur'] = self.config['last_word_dur']
        dataframe['trial_dur_correct'] = (dataframe['word_count'] * dataframe['stem_word_dur']) + self.config['iti_dur'] + dataframe['last_word_dur']
        dataframe['display_trial_feedback'] = self.config['display_trial_feedback']
        dataframe['replace_stimuli'] = self.config['replace']
        dataframe['feedback_type'] = self.config['feedback_type']
        dataframe['target_score'] = self.config['target_score']

        dataframe.drop({'full_sentence'}, inplace=True, axis=1)

        return dataframe

    def _add_random_word(self, dataframe, columns):
        """ sample `frac_random` and add to `full_sentence`
            Args:
                dataframe (pandas dataframe): dataframe
            Returns:
                dataframe with modified `full_sentence` col
        """
        idx = dataframe.groupby(columns).apply(lambda x: x.sample(frac=self.config['frac'], replace=False, random_state=self.random_state)).index

        sampidx = idx.get_level_values(len(columns)) # get third level
        dataframe["trial_type"] = ~dataframe.index.isin(sampidx)
        dataframe["last_word"] = dataframe.apply(lambda x: x["random_word"] if not x["trial_type"] else x["target_word"], axis=1)

        # shuffle the order of the trials
        dataframe = dataframe.apply(lambda x: x.sample(n=self.num_trials, random_state=self.random_state, replace=False)).reset_index(drop=True)

        return dataframe

    def _balance_design(self, dataframe):

        # group the dataframe according to `balance_blocks`
        dataframe = dataframe.groupby([*self.config['balance_blocks']], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.config['replace'])).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        num_stims = int(self.num_trials / len(self.config['balance_blocks']['condition_name']))
        dataframe = dataframe.groupby('condition_name', as_index=False).apply(lambda x: x.sample(n=num_stims, random_state=self.random_state, replace=False)).reset_index(drop=True)

        return dataframe

    def _get_block_info(self):
        # num of blocks (i.e. target files) to make
        self.num_blocks = self.config['num_runs'] * self.config['tile_run']

        # get overall number of trials
        self.num_trials = int(self.config['block_dur_secs'] / (self.config['trial_dur'] + self.config['iti_dur']))

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.config['balance_blocks'].values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int

    def make_targetfile(self):
        """
        makes target file(s) for action observation

        """
        # get info about block
        self._get_block_info()

        # return logging file
        fpath = os.path.join(Defaults.STIM_DIR, self.config['block_name'], self.config['logging_file'])

        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        # filter dataframe
        df_filtered = self._filter_dataframe(dataframe=df)

        # create new columns (`condition_name` etc)
        df_filtered = self._create_new_columns(dataframe=df_filtered)

        seeds = np.arange(self.num_blocks)+1

        # for self.block, self.key in enumerate(self.block_design):
        for self.block in np.arange(self.num_blocks):
            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # balance the dataframe by `condition_name` and `player_num`
            df_target = self._balance_design(dataframe=df_filtered)

            # remove `df_target` rows from the main dataframe so that we're always sampling from unique rows
            if self.config['replace']==False:
                df_filtered = df_filtered.merge(df_target, how='left', indicator=True)
                df_filtered = df_filtered[df_filtered['_merge'] == 'left_only'].drop('_merge', axis=1)

            # add random word based on `self.frac`
            df_target = self._add_random_word(dataframe=df_target, columns=['condition_name']) # 'CoRT_descript'

            # save out target files
            self.TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.config['block_name'])
            self._save_target_files(df_target)

class ActionObservation(MakeFiles):
    """
        This class makes target files for Action Observation using parameters from config file
        Args:
            target_config (dict): dictionary loaded from `action_observation_config.json`
        Kwargs:
            block_name (str): 'rest'
            logging_file (str): csv file containing info about stimuli
            video_name (list of str): name of video(s) to include
            manipulation (str): 'left_right' or 'miss_goal'
            resized (bool): resize frames of video
            balance_blocks (dict): keys are 'player_name', 'condition_name', 'trial_type'
            trial_dur (int): length of trial (sec)
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for block_names (sec)
            hand (str): response hand
            replace (bool): sample stim with or without replacement
            display_trial_feedback (bool): display trial-by-trial feedback
    """

    def __init__(self, target_config, **kwargs):
        super().__init__()
        self.config.update(target_config)
        self.config.update(**kwargs)

    def _filter_dataframe(self, dataframe):

        def _get_player(x):
            if x.find('DC')>=0:
                player_name = 'DC'
            elif x.find('FI')>=0:
                player_name = 'FI'
            elif x.find('EW')>=0:
                player_name = 'EW'
            else:
                print('player does not exist')
            return player_name

        # remove all filenames where any of the videos have not been extracted
        # and where the player did not accurately hit the target (success=F)
        stims_to_remove = dataframe.query('extracted==False or player_success=="?"')["video_name"].to_list()
        df_filtered = dataframe[~dataframe["video_name"].isin(stims_to_remove)]

        # remove rows without video info
        video_name = self.config['video_name']
        df_filtered = df_filtered.query(f'condition_name=={video_name}')

        # create `player_name`
        df_filtered['player_name'] = df_filtered['video_name'].apply(lambda x: _get_player(x))

        # filter `player_name`
        cond = self.config['balance_blocks']['player_name']
        df_filtered = df_filtered.query(f'player_name=={cond}')

        # figure out the actual hits. certain trials (~14%) were misses. enter the actual hit.
        df_filtered.loc[df_filtered['hit_target'].isnull(), 'hit_target'] = df_filtered['instructed_target']

        return df_filtered

    def _create_new_columns(self, dataframe):

        def _get_condition(x):
            if self.config['manipulation']=="miss_goal":
                easy = [1,2,7,8,9,10,15,16]
                hard = [3,4,5,6,11,12,13,14]
            elif self.config['manipulation']=="left_right":
                easy = [1,2,3,4,13,14,15,16]
                hard = [5,6,7,8,9,10,11,12]
            else:
                print('manipulation does not exist')

            if x in easy:
                condition = "easy"
            elif x in hard:
                condition = "hard"
            else:
                condition = float("NaN")
                print(f'{x} not in list')

            return condition

        def _get_trial_type(x):
            if self.config['manipulation']=="miss_goal":
                list1= [5,6,7,8,9,10,11,12]
                list2 = [1,2,3,4,13,14,15,16]
                value1 = "goal"
                value2 = "miss"
            elif self.config['manipulation']=="left_right":
                list1 = [1,2,3,4,5,6,7,8]
                list2 = [9,10,11,12,13,14,15,16]
                value1 = True # 'right'
                value2 = False # 'left'
            else:
                print('manipulation does not exist')

            if x in list1:
                trial = value1
            elif x in list2:
                trial = value2
            else:
                trial = float("NaN")
                print(f'{x} not in list')

            return trial

        # make new image column
        if self.config['resized']:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '_resized' + '.mp4'
        else:
            dataframe['stim'] = dataframe['video_name'] + '_' + dataframe['condition_name'] + '.mp4'

        # divide targets between easy and hard
        dataframe['condition_name'] = dataframe['hit_target'].apply(lambda x: _get_condition(x))

        # either miss_goal or left_right based on manipulation
        dataframe['trial_type'] = dataframe['hit_target'].apply(lambda x: _get_trial_type(x))

        # get time of extraction for video (round to two decimals)
        dataframe['video_start_time'] = np.round(dataframe['interact_start'] - dataframe['secs_before_interact'], 2)
        dataframe['video_end_time'] = np.round(dataframe['interact_start'] + dataframe['secs_after_interact'], 2)

        dataframe['display_trial_feedback'] = self.config['display_trial_feedback']
        dataframe['replace_stimuli'] = self.config['replace']
        dataframe['feedback_type'] = self.config['feedback_type']
        dataframe['target_score'] = self.config['target_score']

        return dataframe

    def _balance_design(self, dataframe):

        # group the dataframe according to `balance_blocks`
        dataframe = dataframe.groupby([*self.config['balance_blocks']], as_index=False).apply(lambda x: x.sample(n=self.num_stims, random_state=self.random_state, replace=self.config['replace'])).reset_index(drop=True)

        # ensure that only `num_trials` are sampled
        num_stims = int(self.num_trials / len(self.config['balance_blocks']['condition_name']))
        dataframe = dataframe.groupby('condition_name', as_index=False).apply(lambda x: x.sample(n=num_stims, random_state=self.random_state, replace=False)).reset_index(drop=True)

        # shuffle the order of the trials
        dataframe = dataframe.apply(lambda x: x.sample(n=self.num_trials, random_state=self.random_state, replace=False)).reset_index(drop=True)

        return dataframe

    def _get_block_info(self):
        # num of blocks (i.e. target files) to make
        self.num_blocks = self.config['num_runs'] * self.config['tile_run']

        # get overall number of trials
        self.num_trials = int(self.config['block_dur_secs'] / (self.config['trial_dur'] + self.config['iti_dur']))

        # get `num_stims` - lowest denominator across `balance_blocks`
        denominator = np.prod([len(stim) for stim in [*self.config['balance_blocks'].values()]])
        self.num_stims = ceil(self.num_trials / denominator) # round up to nearest int

    def make_targetfile(self):
        """
        makes target file(s) for action observation

        """
        # get info about block
        self._get_block_info()

        # return logging file
        fpath = os.path.join(Defaults.STIM_DIR, self.config['block_name'], self.config['logging_file'])

        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        # filter dataframe
        df_filtered = self._filter_dataframe(dataframe=df)

        # create new columns (`trial_type` etc)
        df_filtered = self._create_new_columns(dataframe=df_filtered)

        seeds = np.arange(self.num_blocks)+1

        # for self.block, self.key in enumerate(self.block_design):
        for self.block in np.arange(self.num_blocks):
            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # balance the dataframe by `condition_name` and `player_num`
            df_target = self._balance_design(dataframe=df_filtered)

            # remove `df_target` rows from the main dataframe so that we're always sampling from unique rows
            if self.config['replace']==False:
                df_filtered = df_filtered.merge(df_target, how='left', indicator=True)
                df_filtered = df_filtered[df_filtered['_merge'] == 'left_only'].drop('_merge', axis=1)

            self.TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.config['block_name'])
            self._save_target_files(df_target)

class Rest(MakeFiles):
    """
        This class makes target files for Rest using parameters from config file
        Args:
            block_name (str): 'rest'
            rest_dur_secs (int): length of rest (sec), 0 if no rest
            iti_dur (iti): length of iti (sec)
            instruct_dur (int): length of instruct for block_names (sec)
            hand (str): response hand
            num_trials (int): number of trials per block. default is 1
            display_trial_feedback (bool): display trial-by-trial feedback
    """
    def __init__(self, target_config, **kwargs):
        super().__init__()
        self.config.update(target_config)
        self.config.update(**kwargs)

    def _get_block_info(self):
        # length (in secs) of the block
        self.trial_dur = self.config['rest_dur_secs']

    def _create_new_columns(self):
        start_time = np.round(np.arange(0, self.config['num_trials']*(self.trial_dur+self.config['iti_dur']), self.trial_dur+self.config['iti_dur']), 1)
        data = {"stim": 'fixation', "trial_dur":self.trial_dur, "iti_dur":self.config['iti_dur'], "start_time":start_time, "hand": self.config['hand']}
        dataframe = pd.DataFrame.from_records(data)
        dataframe['display_trial_feedback'] = self.config['display_trial_feedback']
        dataframe['replace_stimuli'] = self.config['replace']
        dataframe['feedback_type'] = self.config['feedback_type']

        return dataframe

    def make_targetfile(self):
        """
        makes target file(s) for rest given parameters in __init__
        """
        # get info about block
        self._get_block_info()

        # save target file
        self.target_name = self.config['block_name'] + '_' + str(self.config['rest_dur_secs']) + 'sec.tsv'

        # create dataframe
        dataframe = self._create_new_columns()
        TARGET_DIR = os.path.join(Defaults.TARGET_DIR, self.config['block_name'])
        if not os.path.exists(TARGET_DIR):
            os.makedirs(TARGET_DIR)
        dataframe.to_csv(os.path.join(TARGET_DIR, self.target_name), index=False, header=True)

        return self.target_name

TASK_MAP = {
    "visual_search": VisualSearch,
    "n_back": NBack,
    "social_prediction": SocialPrediction,
    "semantic_prediction": SemanticPrediction,
    "action_observation": ActionObservation,
    "rest": Rest,
    }

# make target and run files
# mk = MakeFiles()
# # mk.make_all()

# mk.make_runfiles()