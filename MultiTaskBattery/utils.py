# Created 2023: Bassel Arafat, Jorn Diedrichsen, Ince Hussain
import os
import pandas as pd

task_table_name = os.path.dirname(__file__) + '/task_table.tsv'  # where the experiment code is stored
task_table = pd.read_csv(task_table_name, sep = '\t')
tasks_without_run_number = ['n_back', 'verb_generation', 'rest', 'tongue_movement',
                            'oddball', 'demand_grid', 'finger_sequence', 'flexion_extension',
                            'visual_search']

def dircheck(path2dir):
    """
    Checks if a directory exists! if it does not exist, it creates it
    Args:
        dir_path (str, path)
            path to the directory you want to be created
    """
    if not os.path.exists(path2dir):
        print(f"creating {path2dir}")
        os.makedirs(path2dir)

def append_data_to_file(filename,data):
    """ Appends a data frame to an (possibly) existing tsv file
    Args:
        filename (str):
            path to the file
        data (dataframe):
            data to be appended to the file
    """
    if os.path.isfile(filename):
        old_data = pd.read_csv(filename, sep = '\t')
        data = pd.concat([old_data,data],axis = 0)
    data.to_csv(filename, sep = '\t', index = False)