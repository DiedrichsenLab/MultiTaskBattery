# Pontine 7T experiment - main script
# Ladan Shahshahani, Jorn Diedrichsen, Ince Hussain, 2021-23

import sys
import experiment_code.experiment_block as exp_block
import experiment_code.make_target as make_target
from experiment_code.ttl_clock import TTLClock
import constants as const
import experiment_code.utils as ut

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
    # main(sys.argv[1])
    main('test')