from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import time
import random
import seaborn as sns
import glob

from experiment_code.pilot.constants import Defaults

def _concat_subjects(study_name, task_name):
    """
    concatenates all subject data for a given task into one dataframe 

    study_name: 'pilot'
    task_name: any of the following: 'visual_search', 'n_back','social_prediction', 'verb_generation'

    returns dataframes for target file and run file data concatenated across subjects
    """
    subj_dirs = glob.glob(f'{Defaults.RAW_DIR}/{task_name}/s*')

    tf_results_all_subjects = pd.DataFrame()
    rf_results_all_subjects = pd.DataFrame()

    for subj_dir in subj_dirs:
        
        subj_id = subj_dir.split('/')[-1]

        tf_file = os.path.join(subj_dir, f"{study_name}_{subj_id}_{task_name}.csv")
        rf_file = os.path.join(subj_dir, f"{study_name}_{subj_id}.csv")

        # load results
        tf_results = pd.read_csv(tf_file)
        rf_results = pd.read_csv(rf_file)

        # add subj ID column
        tf_results['subj_id'] = subj_id
        rf_results['subj_id'] = subj_id

        tf_results_all_subjects = tf_results_all_subjects.append(tf_results)
        rf_results_all_subjects = rf_results_all_subjects.append(rf_results)

    # do some clean up
    tf_results_all_subjects.reset_index(level=0, inplace=True)

    return tf_results_all_subjects, rf_results_all_subjects

def _add_condition_label(csv_filename, task_name):
    tf_results = pd.read_csv(csv_filename)

    def _get_condition(x):
        if x <= 0.4:
            return 'Hard'
        elif 0.4 < x < 0.6:
            return 'Medium'
        elif x >= 0.6:
            return 'Easy'
        else:
            return None
        # if x < 0.4:
        #     return 'hard'
        # elif x > 0.7:
        #     return 'easy'
        # else:
        #     return None

    # make new condition column called 'condition_label' (specifically for visual_search)
    if task_name == 'visual_search':
        tf_results['condition_label'] = tf_results['condition'].map({4: 'easy', 12: 'hard'})
    elif task_name == 'verb_generation':
        tf_results['condition_label'] = tf_results['cloze_speech_to_text'].apply(_get_condition)
    else:
        tf_results['condition_label'] = tf_results['condition']

    return tf_results[['condition_label']]

def _add_rt_cloze_verb_generation(csv_filename, method):
    tf_results = pd.read_csv(csv_filename)

    tf_results['resp_made'] = tf_results['rt'].apply(lambda x: True if x > 0 else False)
    tf_results['corr_resp'] = tf_results['resp_made'].apply(lambda x: True if x==True else False)

    # load cloze probability
    df_cloze = pd.read_csv(os.path.join(Defaults.RAW_DIR, "verb_generation", f"cloze_probability_{method}.csv"))

     # remove superfluous cols that result from merge
    cols = [c for c in tf_results.columns if bool(re.search(r'\d', c))]
    tf_results = tf_results.drop(cols, axis=1)

    # merge dataframes to get cloze probabilities (drop duplicate columns after merge)
    tf_results = tf_results.merge(df_cloze, left_on='stim_file', right_on='noun').rename({'cloze_prob': f"cloze_{method}"}, axis=1)

    # remove superfluous cols that result from merge
    cols = [c for c in tf_results.columns if bool(re.search(r'\d', c))]
    tf_results = tf_results.drop(cols, axis=1)

    # write out new file
    tf_results.to_csv(csv_filename, index=False)

def _task_specific_preprocess(study_name, task_name):
    subj_dirs = glob.glob(f'{Defaults.RAW_DIR}/{task_name}/s*')

    for subj_dir in subj_dirs:
        
        subj_id = subj_dir.split('/')[-1]

        csv_filename = os.path.join(subj_dir, f"{study_name}_{subj_id}_{task_name}.csv")

        # add rt and cloze to verb generation
        if task_name == 'verb_generation':
            method = 'speech_to_text'
            add_rt_cloze_verb_generation(csv_filename, method)

        # load in filename
        tf_results = pd.read_csv(csv_filename)

        if 'condition_label' in tf_results.columns:
            tf_results = tf_results.drop('condition_label', axis=1)

        # add new condition label to all tasks
        tf_results = pd.concat([tf_results, _add_condition_label(csv_filename, task_name)], axis=1)

        # remove superfluous cols that may have resulted from merge
        cols = [c for c in tf_results.columns if bool(re.search(r'\d', c))]
        tf_results = tf_results.drop(cols, axis=1)

        tf_results.to_csv(csv_filename, index=False)

def _add_excel_info(target_file_all_subjects, run_file_all_subjects):
    """
    adds info from excel sheet about task version, subject id etc

    study_name: 'pilot'
    task_name: any of the following: 'visual_search', 'n_back','social_prediction', 'verb_generation'
    target_file_results: output from concat_subjects
    run_file_results: output from concat subjects

    returns dataframes for target file and run file data appended with participant info
    """
    # load in participant info
    part_info = pd.read_excel(str(Defaults.RAW_DIR / "Participant_Info.xlsx"), col=0)

    # add the version details from participant info to the dataframes
    tf_all_subjects_info = target_file_all_subjects.merge(part_info[['Versions', 'Subject ID']], left_on='subj_id', right_on='Subject ID').drop('Subject ID',axis=1)
    rf_all_subjects_info = run_file_all_subjects.merge(part_info[['Versions', 'Subject ID']], left_on='subj_id', right_on='Subject ID').drop('Subject ID',axis=1)

    return tf_all_subjects_info, rf_all_subjects_info

def clean_data(study_name='pilot', task_name='visual_search'):
    """
    cleans data; calls following functions: 'task_specific_preprocess', 'concat_subjects' and 'add_excel_info'

    study_name: 'pilot'
    task_name: any of the following: 'visual_search', 'n_back','social_prediction', 'verb_generation'

    returns cleaned dataframes for target file and run file data (task specific preprocess, concat across subjs, and part info appended)
    """

    # _task_specific_preprocess(study_name, task_name)

    target_file_all_subjects, run_file_all_subjects = _concat_subjects(study_name, task_name)

    tf_all_subjects_info, rf_all_subjects_info = _add_excel_info(target_file_all_subjects, run_file_all_subjects)

    return tf_all_subjects_info, rf_all_subjects_info