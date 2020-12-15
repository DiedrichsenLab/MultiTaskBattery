import numpy as np
import pandas as pd
import cv2 
import re
import os
import shutil
from math import ceil
import seaborn as sns
from matplotlib import pyplot as plt
import datetime as dt

from experiment_code.pilot.constants import Defaults
from experiment_code.pilot import preprocess_behav_gorilla

import warnings
warnings.filterwarnings("ignore")

class ActionObservation:

    def __init__(self):
        self.task_name = 'action_observation'
        self.versions = [2,3,5,6] #2,3,4,5,6,7,8,9,10
        self.players = ['DC', 'EW', 'FI']
        self.cutoff = 30
        self.player_name = True

    def run_accuracy_version(self, dataframe):
        # accuracy across different levels
        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='block_num', y='Correct', hue='version_descript', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={self.versions}'))
        plt.xlabel('Run', fontsize=20)
        plt.ylabel('% Correct', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()
            
    def run_reaction_time_version(self, dataframe):
        # rt for different levels
        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='block_num', y='rt', hue='version_descript', data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==[True] and version=={self.versions}'))
        plt.xlabel('Run', fontsize=20)
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()
    
    def run_accuracy_trial_type(self, dataframe):
        # rt for different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))

        for i, version in enumerate(versions):

            ax = fig.add_subplot(grid_size, grid_size, i+1)

            try: 
                sns.factorplot(x='block_num',
                        y='Correct', hue='trial_type', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==[True] and version=={version}'),
                        ax=ax) 
            except:
                sns.lineplot(x='block_num',
                            y='Correct', hue='trial_type', 
                            data=dataframe.query(f'Attempt==1 and good_subjs==[True] and version=={version}'),
                            ax=ax) 
            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.legend(loc='lower right', fontsize=6)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('Accuracy', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
    
    def run_accuracy_trial_type_condition(self, dataframe):
        # rt for different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        dataframe['trial_type_condition'] = dataframe['trial_type'] + '-' + dataframe['condition_name']

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))

        for i, version in enumerate(versions):

            ax = fig.add_subplot(grid_size, grid_size, i+1)

            try: 
                sns.factorplot(x='block_num',
                        y='Correct', 
                        hue='trial_type_condition', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==[True] and version=={version}'),
                        ax=ax) 
            except:
                sns.lineplot(x='block_num',
                            y='Correct', 
                            hue='trial_type_condition', 
                            data=dataframe.query(f'Attempt==1 and good_subjs==[True] and version=={version}'),
                            ax=ax) 
            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.legend(loc='lower right', fontsize=6)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('Accuracy', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
    
    def run_reaction_time_trial_type(self, dataframe):
        # rt for different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))

        for i, version in enumerate(versions):

            ax = fig.add_subplot(grid_size, grid_size, i+1)

            sns.lineplot(x='block_num',
                        y='rt', hue='trial_type', 
                        data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==[True] and version=={version}'),
                        ax=ax) 
            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.legend(loc='lower right', fontsize=6)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('Reaction Time', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
    
    def run_accuracy_condition(self, dataframe):
        # accuracy across different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))
       
        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))
        
        for i, version in enumerate(versions):
            
            ax = fig.add_subplot(grid_size, grid_size, i+1)

            try: 
                sns.factorplot(x='block_num', 
                            y='Correct', 
                            hue='condition_name', 
                            data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version} and player_name=={self.players}'),
                            ax=ax)
            except: 
                sns.lineplot(x='block_num', 
                        y='Correct', 
                        hue='condition_name', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version} and player_name=={self.players}'),
                        ax=ax)

            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.legend(loc='lower right', fontsize=6)
            # ax.set_ylim(bottom=.7, top=1.0)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('% Correct', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
   
    def run_reaction_time_condition(self, dataframe):
        # rt for different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))

        for i, version in enumerate(versions):

            ax = fig.add_subplot(grid_size, grid_size, i+1)

            sns.lineplot(x='block_num',
                        y='rt', hue='condition_name', 
                        data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==[True] and version=={version}'),
                        ax=ax) 
            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.legend(loc='lower right', fontsize=6)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('Reaction Time', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
    
    def run_accuracy_player(self, dataframe):
        # accuracy across different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))
       
        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))
        
        for i, version in enumerate(versions):
            
            ax = fig.add_subplot(grid_size, grid_size, i+1)

            try: 
                sns.factorplot(x='block_num', 
                            y='Correct', 
                            hue='player_name', 
                            data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'),
                            ax=ax)
            except:
                sns.lineplot(x='block_num', 
                        y='Correct', 
                        hue='player_name', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'),
                        ax=ax)

            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.legend(loc='lower right', fontsize=6)
            # ax.set_ylim(bottom=.7, top=1.0)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('% Correct', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
    
    def run_accuracy_player_condition(self, dataframe):
        # accuracy across different levels
        fig = plt.figure(figsize=(10,10))

        dataframe['player_condition_name'] = dataframe['player_name'] + '_' + dataframe['condition_name']
       
        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))
        
        for i, version in enumerate(versions):
            
            ax = fig.add_subplot(grid_size, grid_size, i+1)

            try: 
                sns.factorplot(x='block_num', 
                            y='Correct', 
                            hue='player_condition_name', 
                            data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'),
                            ax=ax)
            except:
                sns.lineplot(x='block_num', 
                        y='Correct', 
                        hue='player_name', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'),
                        ax=ax)

            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.legend(loc='lower right', fontsize=6)
            # ax.set_ylim(bottom=.7, top=1.0)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('% Correct', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
    
    def accuracy_condition_version(self, dataframe):
        # accuracy for different levels
        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='condition_name', y='Correct', hue='version_descript', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={self.versions}'))
        plt.xlabel('')
        plt.ylabel('% Correct', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()
    
    def reaction_time_condition_version(self, dataframe):
        # accuracy for different levels
        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='condition_name', y='rt', hue='version_descript', data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==True and version=={self.versions}'))
        plt.xlabel('')
        plt.ylabel('% Correct', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show() 
   
    def separate_condition_version(self, dataframe):
        fig = plt.figure(figsize=(10,10))

        condition_names = dataframe['condition_name'].unique()

        for i, condition_name in enumerate(condition_names):

            ax = fig.add_subplot(1, len(condition_names), i+1)

            sns.lineplot(x='hit_target', 
                        y='Correct', 
                        hue='version_descript', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==True and condition_name=="{condition_name}" and version=={self.versions}'),
                        ax=ax)
            ax.set_title(f'{condition_name} trials', fontsize=20);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
            ax.legend(loc='lower right', fontsize=6)
            # plt.ylim(0.2, 1.0)

        ax.set_xlabel('Targets', fontsize=20)
        ax.set_ylabel('% Correct', fontsize=20)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
        
    def separate_player_version(self, dataframe):
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))

        for i, version in enumerate(versions):

            ax = fig.add_subplot(grid_size, grid_size, i+1)

            sns.factorplot(x='hit_target', 
                        y='Correct', 
                        hue='player_name', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'), # and condition_name=="{condition_name}"
                        ax=ax)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
            ax.legend(loc='lower right', fontsize=6)
            # plt.ylim(0.2, 1.0)

        # ax.set_xlabel('Targets', fontsize=20)
        # ax.set_ylabel('% Correct', fontsize=20)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
   
    def item_analysis(self, dataframe, hit_target=7):
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        grid_size = ceil((len(versions)/2))

        for i, version in enumerate(versions):

            ax = fig.add_subplot(grid_size, grid_size, i+1)

            sns.factorplot(x='Trial Number', 
                        y='Correct', 
                        data=dataframe.query(f'Attempt==1 and good_subjs==True and version==[{version}] and hit_target=={hit_target}'),
                        ci=None,
                        size=10,
                        aspect=2,
                        ax=ax)

            ax.set_xlabel('Trials', fontsize=15),
            ax.set_ylabel('Reaction Time', fontsize=15)
            ax.set_title(f'version {version_descripts[i]}-Target {hit_target}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
        plt.show()
    
    def load_dataframe(self):
        # load in cleaned data
        df = preprocess_behav_gorilla.clean_data(task_name=self.task_name, 
                                            versions=self.versions, 
                                            cutoff=self.cutoff,
                                            player_name=self.player_name)
        return df

class SocialPrediction:

    def __init__(self):
        self.task_name = 'social_prediction'
        self.versions = [4]
        self.cutoff = 30
        self.correct_interaction = True

    def run_accuracy_version(self, dataframe):
        # accuracy across different levels
        sns.set(rc={'figure.figsize':(20,10)})
        # versions = dataframe['version'].unique()

        sns.factorplot(x='block_num', y='Correct', hue='version_descript', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={self.versions}'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('% Correct', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def run_reaction_time_version(self, dataframe):
        # rt for different levels
        sns.set(rc={'figure.figsize':(20,10)})
        # versions = dataframe['version'].unique()

        sns.factorplot(x='block_num', y='rt', hue='version', data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==[True] and version=={self.versions}'))
        plt.xlabel('Run', fontsize=20)
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def run_accuracy_actors(self, dataframe):
        # accuracy for different levels
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            sns.factorplot(x='block_num', y='Correct', hue='actors', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()
    
    def run_accuracy_initiator(self, dataframe):
        # accuracy for different levels
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            sns.factorplot(x='block_num', y='Correct', hue='initiator', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()
    
    def run_accuracy_conditions(self, dataframe):
        # accuracy for different levels
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            sns.factorplot(x='block_num', y='Correct', hue='condition_name', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()

    def run_reaction_time_conditions(self, dataframe):
        # rt for different levels
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            sns.factorplot(x='block_num', y='rt', hue='condition_name', data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('Reaction Time', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()

    def run_accuracy_interact_type(self, dataframe):
        # accuracy for different levels across `interact_type`
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            sns.factorplot(x='block_num', y='Correct', hue='interact_type', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()

    def run_accuracy_interaction(self, dataframe):
        # accuracy for different levels across `interaction`
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            sns.factorplot(x='block_num', y='Correct', hue='interaction', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()

    def accuracy_interaction(self, dataframe):
        # accuracy for different levels across `interaction`
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            sns.factorplot(x='interaction', y='Correct', hue='interaction', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Interactions', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()
    
    def accuracy_interact_type(self, dataframe):
        # accuracy for different levels across `interaction`
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            sns.factorplot(x='interact_type', y='Correct', hue='interaction', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Interaction types', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            # plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()
    
    def summary_accuracy_interact_type(self, dataframe):
        # accuracy for different levels across `interaction`
        sns.set(rc={'figure.figsize':(15,5)})

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):
            plt.figure()
            sns.barplot(x='condition_name', y='Correct', hue='interact_type', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={version}'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('% Correct', fontsize=20)
            plt.title(f'{version_descripts[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()

    def load_dataframe(self):
        # load in cleaned data
        df = preprocess_behav_gorilla.clean_data(task_name=self.task_name, 
                                            versions=self.versions, 
                                            cutoff=self.cutoff,
                                            correct_interaction=self.correct_interaction)

        return df

class SemanticPrediction:

    def __init__(self):
        self.task_name = 'semantic_prediction'
        self.versions = [2, 3, 4]
        self.cutoff = 30
        self.trial_type = True

    def load_dataframe(self):
        # load in cleaned data
        df = preprocess_behav_gorilla.clean_data(task_name=self.task_name, 
                                            versions=self.versions, 
                                            cutoff=self.cutoff,
                                            trial_type=self.trial_type)
        
        return df

    def run_accuracy_version(self, dataframe):
        # accuracy across different levels
        sns.set(rc={'figure.figsize':(20,10)})
        # versions = dataframe['version'].unique()

        sns.factorplot(x='block_num', y='Correct', hue='version', data=dataframe.query(f'Attempt==1 and good_subjs==True and version=={self.versions}'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('% Correct', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()
    
    def run_reaction_time_version(self, dataframe):
        # rt for different levels
        sns.set(rc={'figure.figsize':(20,10)})
        # versions = dataframe['version'].unique()

        sns.factorplot(x='block_num', y='rt', hue='version', data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==[True] and version=={self.versions}'))
        plt.xlabel('Run', fontsize=20)
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show() 
    
    def run_accuracy_condition(self, dataframe):
        # accuracy across different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))
       
        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()
        
        for i, version in enumerate(versions):
            
            ax = fig.add_subplot(1, len(versions), i+1)

            sns.lineplot(x='block_num', 
                        y='Correct', 
                        hue='condition_name', 
                        data=dataframe.query(f'Attempt==1 and trial_type=="meaningful" and good_subjs==True and version=={version}'),
                        ax=ax)

            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.set_ylim(bottom=.7, top=1.0)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('% Correct', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)
        plt.show()

    def run_reaction_time_condition(self, dataframe):
        # rt for different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            ax = fig.add_subplot(1, len(versions), i+1)

            sns.lineplot(x='block_num',
                        y='rt', hue='condition_name', 
                        data=dataframe.query(f'Attempt==1 and Correct==1 and trial_type=="meaningful" and good_subjs==[True] and version=={version}'),
                        ax=ax) 

            ax.set_xlabel('', fontsize=15),
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('Reaction Time', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)
        plt.show()

    def run_reaction_time_trialtype(self, dataframe):
        # rt for different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))
        
        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()
        
        for i, version in enumerate(versions):

            ax = fig.add_subplot(1, len(versions), i+1)

            sns.lineplot(x='block_num', 
                        y='rt', hue='trial_type', 
                        data=dataframe.query(f'Attempt==1 and Correct==1 and good_subjs==[True] and version=={version}'),
                        ax=ax)

            ax.set_xlabel('', fontsize=15)
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            # ax.set_ylim(bottom=.7, top=800)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('Reaction Time', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)
        plt.show()

    def run_reaction_time_cort(self, dataframe):
        # rt for different levels
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            ax = fig.add_subplot(1, len(versions), i+1)

            sns.factorplot(x='block_num', 
                        y='rt', 
                        hue='CoRT_descript', 
                        data=dataframe.query(f'Attempt==1 and Correct==1 and trial_type=="meaningful" and good_subjs==[True] and version=={version}'),
                        ax=ax)

            ax.set_xlabel('', fontsize=15)
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

        ax.set_xlabel('Run', fontsize=15)
        ax.set_ylabel('Reaction Time', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)
        plt.show()

    def cloze_distribution(self, dataframe):
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            fig.add_subplot(1, len(versions), i+1)

            df = dataframe.query(f'version=={version}')
            
            sns.distplot(df['cloze_probability'])
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('cloze probability', fontsize=15)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.show()

    def cort_distribution(self, dataframe):
        # sns.set(rc={'figure.figsize':(20,10)})

        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            fig.add_subplot(1, len(versions), i+1)

            df = dataframe.query(f'version=={version}')

            sns.distplot(df['CoRT_mean'])
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('cort scaling', fontsize=15)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.show()

    def describe_block_design(self, dataframe):
        # sns.set(rc={'figure.figsize':(20,10)})

        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            fig.add_subplot(1, len(versions), i+1)

            sns.countplot(x='block_num', hue='CoRT_descript', data=dataframe.query(f'version=={version} and trial_type=="meaningful"'))
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('block_design', fontsize=15)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.show()

    def distribution_cloze_by_run(self, dataframe):
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()
        runs = dataframe['block_num'].unique()

        # dataframe = dataframe.query('CoRT_descript=="strong CoRT"')

        # loop over versions
        for i, version in enumerate(versions):

            df_version = dataframe.query(f'version=={version}')

            fig.add_subplot(1, len(versions), i+1)

            # loop over runs and plot distribution
            for run in runs:
                sns.kdeplot(df_version.loc[df_version['block_num']==run]['cloze_probability'], shade=True)
                
            # plot stuff        
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('cloze probability', fontsize=15)
            plt.legend(runs, fontsize=10)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

        plt.show()
        
    def distribution_cort_by_run(self, dataframe):
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()
        runs = dataframe['block_num'].unique()

        # dataframe = dataframe.query('CoRT_descript=="strong CoRT"')

        # loop over versions
        for i, version in enumerate(versions):

            df_version = dataframe.query(f'version=={version}')

            fig.add_subplot(1, len(versions), i+1)

            # loop over runs and plot distribution
            for run in runs:
                sns.kdeplot(df_version.loc[df_version['block_num']==run]['CoRT_mean'], shade=True)
                
            # plot stuff        
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('cort scaling', fontsize=15)
            plt.legend(runs, fontsize=10)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

        plt.show()
    
