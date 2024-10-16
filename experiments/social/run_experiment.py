# Social cognition experiment - main script
# created 2024: Caroline Nettekoven

import MultiTaskBattery.experiment_block as exp_block
import constants as const

def main(subj_id):
    """_summary_
    make sure you to adjust constanst.py file before running the experiment
    (e.g., experiment_name, eye_tracker, screen, etc.)

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
    # main('training')
    main('sub-04')