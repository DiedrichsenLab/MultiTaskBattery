# Defines the Experiment as a class
# March 2021: First version: Ladan Shahshahani  - Maedbh King - Suzanne Witt,
# Revised 2023: Bassel Arafat, Jorn Diedrichsen, Ince Hussain

import pandas as pd
import sys

from psychopy import visual, core, gui, event
import MultiTaskBattery.utils as ut
import MultiTaskBattery.task_blocks as tasks
from MultiTaskBattery.ttl_clock import TTLClock
from MultiTaskBattery.screen import Screen
# import pylink as pl # to connect to eyelink


class Experiment:
    """
    A general class with attributes common to experiments
    """

    def __init__(self, const, subj_id):
        """
            const (module):
                local constants.py module (see pontine_7T/constants.py) as example
            subj_id (str):
                id for the subject
        """

        self.exp_name   = const.exp_name
        self.subj_id    = subj_id
        self.run_number = 0
        self.const = const
        self.ttl_clock = TTLClock()
        self.set_const_defaults()

        # open screen and display fixation cross
        ### set the resolution of the subject screen here:
        self.screen = Screen(const.screen)

        # connect to the eyetracker already
        if self.const.eye_tracker:
            import pylink as pl
            # create an Eyelink class
            ## the default ip address is 100.1.1.1.
            ## in the ethernet settings of the laptop,
            ## set the ip address of the EyeLink ethernet connection
            ## to 100.1.1.2 and the subnet mask to 255.255.255.0
            self.tk = pl.EyeLink('100.1.1.1')

    def set_const_defaults(self): # jorn, do we need this?
        """ Make sure all the necessary variables are set in the constant file - otherwise set them to default values"""
        # if not 'stim_dir' in dir(self.const):
        #     self.const.stim_dir = Path(os.path.dirname(os.path.dirname(__file__))) / 'stimuli'  # where the experiment code is stored
        pass

    def confirm_run_info(self):
        """
        Presents a GUI to confirm the settings for the run:

        The following parameters will be set:
        run_number      - run number
        subj_id         - id assigned to the subject
        ttl_flag        - should the program wait for the ttl pulse or not? For scanning THIS FLAG HAS TO BE SET TO TRUE

        Args:
        """
        if not self.const.debug:
            # a dialog box pops up so you can enter info
            #Set up input box
            inputDlg = gui.Dlg(title = f"{self.exp_name}")
            inputDlg.addField('Enter Subject id (str):',initial = self.subj_id)      # run number (int)
            inputDlg.addField('Enter Run Number (int):',initial = self.run_number+1)      # run number (int)
            inputDlg.addField('Run File name (str):',initial = f'run_{self.run_number+1:02d}.tsv')      # run number (int)
            inputDlg.addField('Wait for TTL pulse?', initial = True) # a checkbox for ttl pulse (set it true for scanning)

            inputDlg.show()

            if inputDlg.OK:
                self.subj_id        = str(inputDlg.data[0])
                self.run_number     = int(inputDlg.data[1])
                self.run_filename   = str(inputDlg.data[2])
                self.wait_ttl       = bool(inputDlg.data[3])
            else:
                sys.exit()

        else:
            print("running in debug mode")
            # pass on the values for your debugging with the following keywords
            self.subj_id = 'test00'
            self.run_number = self.run_number+1
            self.run_filename = 'run_01.tsv'
            self.wait_ttl = True

    def init_run(self):
        """initializing the run:
            making sure a directory is created for the behavioral results
            getting run file
            Initializes all the tasks for a run
        """

        # 1. get the run file info: creates self.run_info
        self.run_info = pd.read_csv(self.const.run_dir / self.run_filename,sep='\t')

        # 2. Initialize the all tasks that we need
        self.task_obj_list = [] # a list containing task objects in the run
        for t_num, task_info in self.run_info.iterrows():
            # create a task object for the current task, reads the trial file, and append it to the list
            t = ut.task_table[ut.task_table['name']== task_info.task_name]
            task_info['code'] = t.code
            class_name = t.task_class.iloc[0]
            TaskClass = getattr(tasks, class_name)
            Task_obj  = TaskClass(task_info,
                                 screen = self.screen,
                                 ttl_clock = self.ttl_clock,
                                 const = self.const)
            Task_obj.init_task()
            self.task_obj_list.append(Task_obj)

        # 3. make subject folder in data/raw/<subj_id>
        subj_dir = self.const.data_dir / self.subj_id
        ut.dircheck(subj_dir) # making sure the directory is created!
        self.run_data_file = self.const.data_dir / self.subj_id / f"{self.subj_id}.tsv"


    def run(self):
        """
        run a run of the experiment
        """
        print(f"running the experiment")
        self.screen.fixation_cross()
        self.ttl_clock.reset()
        self.ttl_clock.wait_for_first_ttl(wait = self.wait_ttl)
        run_data = []

        # Start the eyetracker
        if self.const.eye_tracker:
            self.start_eyetracker()

        for t_num, task in enumerate(self.task_obj_list):
            print(f"Starting: {task.name}")

            # Take the task data from the run_info dataframe
            r_data = self.run_info.iloc[t_num].copy()

            # wait till it's time to start the task
            r_data['real_start_time'],r_data['start_ttl'],r_data['start_ttl_time'] = self.ttl_clock.wait_until(r_data.start_time)

            ## sending a message to the edf file specifying task name
            if self.const.eye_tracker:
                pl.sendMessageToFile(f"task_name: {task.name} start_track: {pl.currentUsec()} real start time {r_data.real_start_time} TR count {ttl_clock.ttl_count}")

            # display the instruction text for the task. (instructions are task specific)
            task.display_instructions()

            # wait for a time period equal to instruction duration
            self.ttl_clock.wait_until(r_data.start_time + r_data.instruction_dur)

            # Run the task (which saves its data to the target)
            task.start_time = self.ttl_clock.get_time()
            r_data['acc'],r_data['rt'] = task.run()

            # Add the end time of the task
            r_data['real_end_time'] = self.ttl_clock.get_time()
            run_data.append(r_data)

        # Wait for the last end time of run
        self.ttl_clock.wait_until(r_data.end_time)

        # Stop the eyetracker
        if self.const.eye_tracker:
            self.stop_eyetracker()
            self.tk.receiveDataFile(self.tk_filename, self.tk_filename)

        # save the run data to the run file
        run_data = pd.DataFrame(run_data)
        run_data.insert(0,'run_num',[self.run_number]*len(run_data))
        ut.append_data_to_file(self.run_data_file, run_data )

        # Save the trial data for each task
        for task in self.task_obj_list:
            task.save_data(self.subj_id, self.run_number)

        # show the scoreboard
        self.display_run_feedback(run_data)


    def display_run_feedback(self, run_data):
        """ Displays a score board for the tasks in the task_list

            Args:
                run_data (pd.DataFrame): a dataframe containing the run data
        """
        score_text = f"Run finished\nTask\tAcc\tRT\n"

        for i,task in enumerate(self.task_obj_list):
            if task.feedback_type!='none':
                score_text += f"{task.name}\t{run_data['acc'][i]:.2f}\t{run_data['rt'][i]:.3f}\n"

        score_display = visual.TextStim(self.screen.window, text=score_text, color=[-1, -1, -1])
        score_display.draw()
        self.screen.window.flip()
        event.waitKeys()

    def start_eyetracker(self):
        """
        sets up a connection with the eyetracker and start recording eye position
        """

        # opening an edf file to store eye recordings
        ## the file name should not have too many characters (<=8?)
        ### get the run number
        self.tk_filename = f"s_{self.run_number}.edf"
        # self.tk_filename = f"{self.subj_id}_r{self.run_number}.edf"
        self.tk.openDataFile(self.tk_filename)
        # set the sampling rate for the eyetracker
        ## you can set it to 500 or 250
        self.tk.sendCommand("sample_rate  500")
        # start eyetracking and send a text message to tag the start of the file
        self.tk.startRecording(1, 1, 1, 1)
        # pl.sendMessageToFile(f"task_name: {self.task_name} start_track: {pl.currentUsec()}")
        return

    def stop_eyetracker(self):
        """
        stop recording
        close edf file
        receive edf file?
            - receiving the edf file takes time and might be problematic during scanning
            maybe it would be better to take the edf files from eyelink host computer afterwards
        """
        self.tk.stopRecording()
        self.tk.closeDataFile()
        # self.tk.receiveDataFile(self.tk_filename, self.tk_filename)
        self.tk.close()
        return

