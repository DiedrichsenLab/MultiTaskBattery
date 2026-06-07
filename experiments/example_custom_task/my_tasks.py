"""Custom Task and TaskFile classes for the example_custom_task experiment.

Each task has both a Task class (e.g. SilentWord) and a matching
TaskFile generator class with a 'File' suffix (e.g. SilentWordFile).

This module is registered in constants.py via `task_modules = [my_tasks]`,
and make_files.py looks up the TaskFile version by appending 'File' to the
class name from task_table.tsv.
"""

import pandas as pd
import numpy as np
from psychopy import visual, event
from MultiTaskBattery.task_blocks import Task
from MultiTaskBattery.task_file import TaskFile


class SilentWord(Task):
    """Show one word per trial. Participant silently reads it. No response."""

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'none'

    def display_instructions(self):
        self.instruction_text = (f"{self.descriptive_name} Task\n\n" "Silently read each word as it appears.")
        instr = visual.TextStim(self.window, text=self.instruction_text, height=self.const.instruction_text_height, color=[-1, -1, -1])
        instr.draw()
        self.window.flip()

    def run_trial(self, trial):
        word = visual.TextStim(self.window, text=trial['word'], color=(-1, -1, -1), height=2)
        word.draw()
        self.window.flip()
        self.ttl_clock.wait_until(self.ttl_clock.get_time() + trial['trial_dur'])
        return trial


class SilentWordFile(TaskFile):
    """Generates trial TSV for SilentWord: picks random words from a small bank."""

    WORDS = ['apple', 'river', 'mountain', 'forest', 'thunder',
             'meadow', 'piano', 'crystal', 'shadow', 'ocean']

    def __init__(self, const):
        super().__init__(const)
        self.name = 'silent_word'

    def make_task_file(self, task_dur=20, trial_dur=2.5, file_name=None):
        n_trials = int(np.floor(task_dur / trial_dur))
        trial_info = []
        t = 0.0
        for n in range(n_trials):
            trial_info.append({
                'trial_num': n,
                'word': np.random.choice(self.WORDS),
                'trial_dur': trial_dur,
                'start_time': t,
                'end_time': t + trial_dur,
                'display_trial_feedback': False,
            })
            t += trial_dur

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return trial_info



class OddEven(Task):
    """Show one digit per trial. Response: key 1 if odd, key 2 if even."""

    def __init__(self, info, screen, ttl_clock, const, subj_id):
        super().__init__(info, screen, ttl_clock, const, subj_id)
        self.feedback_type = 'acc+rt'

    def display_instructions(self):
        odd_key  = self.const.response_keys[0]
        even_key = self.const.response_keys[1]
        self.instruction_text = (
            f"{self.descriptive_name} Task\n\n"
            f"If the digit is ODD, press {odd_key}.\n"
            f"If the digit is EVEN, press {even_key}."
        )
        instr = visual.TextStim(self.window, text=self.instruction_text, height=self.const.instruction_text_height, color=[-1, -1, -1])
        instr.draw()
        self.window.flip()

    def run_trial(self, trial):
        event.clearEvents()
        digit = visual.TextStim(self.window, text=str(trial['digit']), color=(-1, -1, -1), height=3)
        digit.draw()
        self.window.flip()

        trial['response'], trial['rt'] = self.wait_response(self.ttl_clock.get_time(), trial['trial_dur'])
        trial['correct'] = (trial['response'] == trial['correct_response'])
        self.display_trial_feedback(trial['display_trial_feedback'], trial['correct'])
        return trial


class OddEvenFile(TaskFile):
    """Generates trial TSV for OddEven: random digit 1-9 per trial."""

    def __init__(self, const):
        super().__init__(const)
        self.name = 'odd_even'

    def make_task_file(self, task_dur=20, trial_dur=2.0, iti_dur=0.5, file_name=None):
        n_trials = int(np.floor(task_dur / (trial_dur + iti_dur)))
        trial_info = []
        t = 0.0
        for n in range(n_trials):
            digit = int(np.random.randint(1, 10))
            trial_info.append({
                'trial_num': n,
                'digit': digit,
                'correct_response': 1 if digit % 2 == 1 else 2,
                'trial_dur': trial_dur,
                'iti_dur': iti_dur,
                'start_time': t,
                'end_time': t + trial_dur + iti_dur,
                'display_trial_feedback': True,
            })
            t += trial_dur + iti_dur

        trial_info = pd.DataFrame(trial_info)
        if file_name is not None:
            trial_info.to_csv(self.task_dir / self.name / file_name, sep='\t', index=False)
        return trial_info
