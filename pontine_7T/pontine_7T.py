# Defines the Experiment as a class
# @ Ladan Shahshahani June 2021

# import libraries
# from quickstart import main
import experiment_code.experiment_block as exp_block
import experiment_code.make_target as make_target
from experiment_code.task_blocks import TASK_MAP
from experiment_code.ttl import ttl

# 1. first go to constants.py and make sure you have made the following changes:
# experiment_name = 'pontine_7T'
# base_dir = Path(<your chosen path>).absolute()

# 2. make sure your base directory contains the following folder
# 'experiment_code'
# 'stimuli'     -   stimuli folder should contain all the stimuli for the tasks in your experiment

# 3. making sure you have all the necessary folders
# consts.dirtree()


# 4. create target files first (if already not done)
def create_target(task_list = ['visual_search', 'action_observation_knots', 'flexion_extension', 
                              'finger_sequence', 'theory_of_mind', 'n_back', 'semantic_prediction', 
                              'rest', 'romance_movie'], 
                  num_runs = 8):

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
    Custom_Exp = exp_block.Experiment(exp_name="pontine_7T", subj_id='fmri_sim')
    Custom_Exp.simulate_fmri(**kwargs)

# 5. run the experiment.
## change debug to False once you are sure everything is debugged 
## make sure that you have changed the screen_res to the res for the subject screen
## display mode should also be in extend!
def main(subj_id, exp_name = "pontine_7T", debug = True, eye_flag = False):
    # 1. create a class for the experiment
    Custom_Exp = exp_block.Experiment(exp_name=exp_name, subj_id=subj_id, eye_flag=eye_flag)

    # 2. get experiment information
    exp_info = Custom_Exp.set_info(debug = debug)

    # 3. run the run
    Custom_Exp.run()

    return

if __name__ == "__main__":
    main(debug = True)