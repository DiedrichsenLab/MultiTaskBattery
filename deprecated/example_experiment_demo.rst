Example Experiment
========================

For this demo, we will run an experiment using predefined run and task files. You can explore the run and task files in the `task_files` and `run_files` directories to understand what tasks are included in the experiment.

.. image:: images/run_vs_task_file.png
   :width: 20000

Step 1: Constants.py
---------------------
Ensure the `constants.py` file is configured with the required information. For instance, you can modify the screen resolution, response keys, and other settings as needed.

Step 2: run.py
---------------------
Update the `subject_id` input in the main function to match the desired subject ID.

Step 3: Run the Experiment
---------------------------
Execute the script in `run.py`. Output files will be saved in the `data` folder, with the subject ID included in the filename.

Step 4: GUI
---------------------------
A GUI pre-filled with information from `constants.py` and the provided subject ID will appear. Verify the details and click the "Ok" button to start the experiment.

.. image:: images/Run_GUI.png
   :width: 600

If "Wait for TTL Pulse?" is selected in the GUI, the experiment will wait for a TTL pulse before starting. Otherwise, the experiment will begin immediately.

