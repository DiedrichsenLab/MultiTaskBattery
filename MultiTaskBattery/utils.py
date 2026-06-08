# Created 2023: Bassel Arafat, Jorn Diedrichsen, Ince Hussain
import os
import pandas as pd
import MultiTaskBattery.task_blocks as tasks
import MultiTaskBattery.task_file as task_files

# DEPRECATED: kept for backwards compatibility with external user scripts.
# make_files.py no longer consults this list — instead it inspects each task's
# make_task_file signature for a run_number parameter. To opt your task out of
# receiving a run_number, simply omit it from the signature.
tasks_without_run_number = ['n_back', 'verb_generation', 'rest', 'tongue_movement',
                            'oddball', 'demand_grid', 'demand_grid_easy_diff','finger_sequence', 'finger_sequence_surprise', 'flexion_extension',
                            'visual_search', 'serial_reaction_time', 'rest_surprise', 'rest_surprise_images', 'rest_surprise_sound_images','temp_deviant']

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

def get_task_table(exp_dir=None):
    """ Reads the task_table.tsv file from the experimental directory
    and the package direction and concatenates them, avoiding duplicates
    Args:
        exp_dir (str, path, optional):
            path to the experiment directory. If None, returns only the general table.
    Returns:
        task_table (dataframe):
            dataframe containing the task table
    """
    gen_task_table = os.path.dirname(__file__) + '/task_table.tsv'  # where the experiment code is stored
    task_table = pd.read_csv(gen_task_table, sep = '\t')
    if exp_dir is not None:
        exp_task_table = os.path.join(exp_dir, 'task_table.tsv')
        if os.path.isfile(exp_task_table):
            exp_task_table = pd.read_csv(exp_task_table, sep = '\t')
            task_table = pd.concat([task_table, exp_task_table], axis = 0).drop_duplicates(subset='name').reset_index(drop=True)
    return task_table

def get_task_class(const, class_name):
    """ Searches for the task class in the list of task modules and returns it
    Args:
        const (constant object):
            constant.py object containing the list of task modules to search for the task class
        class_name (str):
            name of the task class to be searched for   
    Returns:        
        TaskClass (class):
            the task class that was searched for
    """
    # First tries to find the class in the custom list of task modules
    if hasattr(const, 'task_modules'):
        for module in const.task_modules:
            if hasattr(module, class_name):
                return getattr(module, class_name)
    if hasattr(tasks, class_name):
        return getattr(tasks, class_name)
    else:
        raise NameError(f"Task class {class_name} not found in any of the task modules, make sure to add the module to the list of task_modules in constants.py")

def get_task_file_class(const, class_name):
    """ Searches for the TaskFile class in the list of task modules and returns it.
    Mirrors get_task_class but for the file-generation side. Custom TaskFile
    classes follow a '<class>File' naming convention so they can coexist with
    their matching Task class in the same module (e.g. SilentWord and
    SilentWordFile in my_tasks.py).

    Args:
        const (constant object):
            constants.py object containing the task_modules list
        class_name (str):
            base class name from task_table.tsv (e.g. 'SilentWord')
    Returns:
        TaskFileClass (class):
            the TaskFile class that was searched for
    """
    # First tries '<class>File' in the custom list of task modules
    if hasattr(const, 'task_modules'):
        suffixed = class_name + 'File'
        for module in const.task_modules:
            if hasattr(module, suffixed):
                return getattr(module, suffixed)
    # Otherwise fall back to the framework's task_file module (bare class name)
    if hasattr(task_files, class_name):
        return getattr(task_files, class_name)
    else:
        raise NameError(f"TaskFile class {class_name} (or {class_name}File) not found in any task module, make sure to add the module to task_modules in constants.py")
