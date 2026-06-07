# Main script for example_custom_task
import MultiTaskBattery.experiment_block as exp_block
import constants as const


def main(subj_id):
    """ Main experiment function.
    Ensure constants.py is configured before running (response keys,
    screen settings, eye tracker, etc.).

    Args:
        subj_id (str): Subject ID
    """
    my_Exp = exp_block.Experiment(const, subj_id=subj_id)

    while True:
        my_Exp.confirm_run_info()
        my_Exp.init_run()
        my_Exp.run()
    return


if __name__ == "__main__":
    main('subject-00')
