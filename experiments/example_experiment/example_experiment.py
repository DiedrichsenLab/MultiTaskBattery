# Main Script for an example experiment
# Diedrichsenlab 2021-24

import sys
import MultiTaskBattery.experiment_block as exp_block
import constants as const

def main(subj_id):
    """Main function
    The constanst.py file sets the default - ensure that you have the correct one
    selected before running the experiment.

    Args:
        subj_id (str): id of the subject
    """
    my_Exp = exp_block.Experiment(const, subj_id=subj_id)

    while True:
        my_Exp.confirm_run_info()
        my_Exp.init_run()
        my_Exp.run()
    return

if __name__ == "__main__":
    # main(sys.argv[1])
    main('sub-01')