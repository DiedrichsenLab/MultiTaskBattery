Building new experiments
========================

Make a new project folder at the root of the repository (`experiments/<your_experiment_name>`). 
Then create a constants file.

Constants file
--------------
Specify the parameters and settings for your experiment in a constants file.
This file should be in the root of your project folder (`experiments/<your_experiment_name>`).

The constants file should contain the following variables:

- Experiment Name
- Response Keys, e.g. `['y', 'g', 'r', 'm']` inside the scanner or `['1', '2', '3', '4']` on a Macbook outside the scanner
- Response Fingers, e.g. `['Pinky', 'Ring', 'Middle', 'Index']`
- Path definitions:
  - Experiment Directory (`exp_dir`): Path where the experiment code is stored
  - Task Directory (`task_dir`): Contains target files for the task
  - Run Directory (`run_dir`): Contains run files for each session
  - Data Directory (`data_dir`): Where result files are saved
  - Stimulus Directory (`stim_dir`): Where the stimuli are stored
- Default Run Filename: `'run_{}.tsv'`
- Eye Tracker: whether you will be collecting eye tracking data
- Screen Settings



Generating run and task files
-----------------------------

**Run Files**
Generate run files that specify the structure of the runs, including the order of the tasks for the run, which task file contains the stimuli for this run, etc.

Each run file should contain the following columns:
- task_name: Name of the task
- task_code: short name of the task
- task_file: Name of the task file for this run
- instruction_dur: Duration of the instruction screen before the task starts (in seconds)
- start_time: Start time of the task (in seconds)
- end_time: End time of the task (in seconds)


**Task Files**
Generate task files that specify the structure of the tasks within each run (e.g. the stimuli, the correct response, whether to display feedback, etc.).

Each task file should contain the following columns:

- Key columns, for example in the case of four response keys (e.g. in the RMET task or Finger Sequence task):
  - key_one: Key for the first option
  - key_two: Key for the second option
  - key_three: Key for the third option
  - key_four: Key for the fourth option
- trial_num: Trial number
- hand: Hand used for the task (left or right)
- trial_dur: Duration of the trial (in seconds)
- iti_dur: Inter-trial interval duration (in seconds)
- stim: Stimulus presented
- display_trial_feedback: Whether to display feedback after each trial
- - start_time: Start time of the trial (in seconds)
- end_time: End time of the trial (in seconds)

  
Optional variables, depending on the type of task and the experiment would be, in the case of the RMET task:
- options: Answer options presented on the screen
- condition: Condition of the trial (Detecting emotion or age)
- answer: Correct response


**Example Code**

.. code-block:: python

    tasks = ['n_back','rest'] # ,'social_prediction','verb_generation'
    for r in range(1,9):
        tfiles = [f'n_back_{r:02d}.tsv','rest_30s.tsv'] # f'social_prediction_{r:02d}.tsv',f'verb_generation_{r:02d}.tsv',
        T  = mt.make_run_file(tasks,tfiles)
        T.to_csv(const.run_dir / f'run_{r:02d}.tsv',sep='\t',index=False)

        # for each of the runs, make a target file
        for task,tfile in zip(tasks, tfiles):
            cl = mt.get_task_class(task)
            myTask = getattr(mt,cl)(const)
            myTask.make_trial_file(file_name = tfile)





Writing your experiment function
--------------------------------

After generating the tasks and run files, you can write your own main script to run the experiment.

**Example Code**

.. code-block:: python

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
      main('sub-01')
