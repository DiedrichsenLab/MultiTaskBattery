# Pontine 7T experiment - main script
# Ladan Shahshahani, Jorn Diedrichsen, Ince Hussain, 2021-23

import sys
import experiment_code.experiment_block as exp_block
import experiment_code.make_target as make_target
from experiment_code.ttl_clock import TTLClock
import constants as const
import experiment_code.utils as ut

# 4. create target files first (if already not done)
def create_target(task_list = ['visual_search', 'flexion_extension',
                              'finger_sequence', 'theory_of_mind', 'n_back', 'semantic_prediction',
                              'rest'],
                  num_runs = 8):
    """action_observation_knots', ''romance_movie''"""
    """
    makes target and session files for the pontine experiment.
    Args:
        task_list (list)    -   list containing task names to be included in the experiment
        num_runs (int)      -   number of runs to be created for the task
    """
    ## behavioral
    make_target.make_files(task_list = task_list, study_name='behavioral', num_runs = num_runs)
    ## fmri
    make_target.make_files(task_list = task_list, study_name='fmri', num_runs = num_runs)

def simulate(**kwargs):
    """simulate the fMRI experiment
    will be used to simulate the experiment for debugging purposes
    and getting a sense of how the experiment will look like in the scanner
    """
    Custom_Exp = exp_block.Experiment(exp_name="pontine_7T", subj_id='fmri_sim')
    Custom_Exp.simulate_fmri(**kwargs)

# 5. run the experiment.
def main(subj_id):
    """_summary_
    change debug to False once you are sure everything is debugged
    make sure that you have changed the screen_res to the res for the subject screen
    display mode should also be in extend!

    Args:
        subj_id (str): id of the subject
        debug (bool, optional): Defaults to True for debugging
        eye_flag (bool, optional): Do you want to do the eyetracking?. Defaults to False.
    """
    my_Exp = exp_block.Experiment(const, subj_id=subj_id)

    while True:
        my_Exp.confirm_run_info()
        my_Exp.init_run()
        my_Exp.run()
    return

if __name__ == "__main__":
    main(sys.argv[1])