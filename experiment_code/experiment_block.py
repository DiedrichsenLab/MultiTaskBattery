# Defines the Experiment as a class
# @ Ladan Shahshahani  - Maedbh King June 2021

# import libraries
import os
import pandas as pd
import glob
import sys
from pathlib import Path

from psychopy import visual, core, event, gui # data, logging
import experiment_code.utils as ut
from experiment_code.task_blocks import task_map
from experiment_code.ttl_clock import TTLClock
from experiment_code.screen import Screen
# from psychopy.hardware.emulator import launchScan
from psychopy.hardware import keyboard
# import pylink as pl # to connect to eyelink


class Experiment:
    """
    A general class with attributes common to experiments
    """

    def __init__(self, const, subj_id, **kwargs):
        """
            const (module):
                local constants.py module (see pontine_7T/constants.py) as example
            subj_id (str):
                id for the subject
        """

        self.exp_name   = const.exp_name
        self.subj_id    = subj_id
        self.const = const
        self.ttl_clock = TTLClock()
        self.__dict__.update(kwargs)
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

    def set_const_defaults(self):
        """ Make sure all the necessary variables are set in the constant file - otherwise set them to default values"""
        if self.const.stim_dir is None:
            self.const.stim_dir = Path(os.path.dirname(os.path.dirname(__file__))) / 'stimuli'  # where the experiment code is stored

    def confirm_run_info(self):
        """
        Presents a GUI to confirm the settings for the run:

        The following parameters will be set:
        behav_trianing  - is it behavioral training or scanning?
            ** behavioral training target/run files are always stored under behavioral and scanning files are under fmri
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
            inputDlg.addField('Enter Run Number (int):',initial =1 )      # run number (int)
            inputDlg.addField('Run File name (str):',initial = 'run_01.tsv')      # run number (int)
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
            self.run_number =  1
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
            TaskName = task_map[task_info.task_name]

            Task_obj  = TaskName(task_info,
                                 screen = self.screen,
                                 ttl_clock = self.ttl_clock,
                                 const = self.const)
            Task_obj.init_task()
            self.task_obj_list.append(Task_obj)

        # 3. make subject folder in data/raw/<subj_id>
        subj_dir = self.const.data_dir / self.subj_id
        ut.dircheck(subj_dir) # making sure the directory is created!
        self.run_data_file = self.const.data_dir / self.subj_id / f"subj-{self.subj_id}.tsv"


    def run(self):
        """
        run a run of the experiment
        """
        print(f"running the experiment")
        self.screen.fixation_cross()

        self.ttl_clock.wait_for_first_ttl(wait = self.wait_ttl)
        run_data = []

        # Start the eyetracker
        if self.const.eye_tracker:
            self.start_eyetracker()

        for t_num, task in enumerate(self.task_obj_list):
            print(f"Starting{task.name}")

            # Take the task data from the run_info dataframe
            r_data = self.run_info.iloc[t_num].copy()
            # wait till it's time to start the task

            r_data['real_start_time'],r_data['start_ttl'],r_data['start_ttl_time'] = self.ttl_clock.wait_until(task.start_time)

            ## sending a message to the edf file specifying task name
            if self.const.eye_tracker:
                pl.sendMessageToFile(f"task_name: {task.name} start_track: {pl.currentUsec()} real start time {r_data.real_start_time} TR count {ttl_clock.ttl_count}")

            # display the instruction text for the task. (instructions are task specific)
            task.display_instructions()

            # wait for a time period equal to instruction duration
            self.ttl_clock.wait_until(r_data.start_time + r_data.instruct_dur)

            # Run the task and collect the responses
            task.run()

            r_data['real_end_time'] = self.ttl_clock.get_time()
            run_data.append(r_data)

        ut.append_data_to_file(self.run_data_file, pd.DataFrame(run_data))

        # Stop the eyetracker
        if self.const.eye_tracker:
            self.stop_eyetracker()


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

    def show_scoreboard(self, taskObjs, screen):
        """
        Presents a score board in the end of the run
        Args:
            taskObjs(list)        -   a list containing task objects
            screen                -   screen object for display
        """

        subj_dir = consts.raw_dir/ self.study_name / 'raw' / self.subj_id
        # loop over task objects and get the feedback
        feedback_all = []
        for obj in taskObjs:

            # get the task name
            t_name = obj.name

            # get the response dataframe saved for the task
            dataframe = pd.read_csv(glob.glob(os.path.join(subj_dir , f'*{t_name}*'))[0])

            # get the feedback dictionary for the task
            feedback = obj.get_task_feedback(dataframe, obj.feedback_type)

            # get the corresponding text for the feedback and append it to the overal list
            # feedback_text = f'{t_name}\n\nCurrent score: {feedback["curr"]} {feedback["measure"]}\n\nPrevious score: {feedback["prev"]} {feedback["measure"]}'
            # feedback_all.append(feedback_text)

            feedback_text = f'{t_name}\n\nCurrent score: {feedback["curr"]} {feedback["measure"]}'
            feedback_all.append(feedback_text)

        # display feedback table at the end of the run
        ## position where the feedback for each task will be shown

        positions = [(-9, -6), (0, -6), (9, -6),
                    (-9, 0), (0, 0), (9, 0),
                    (-9, 6), (0, 6), (9, 6)]
        for position, feedback in zip(positions, feedback_all):
            scoreboard = visual.TextStim(screen.window, text = feedback, color = [-1, -1, -1], pos = position, height = 0.7)
            scoreboard.draw()

        screen.window.flip()

        event.waitKeys()
        kb = keyboard.Keyboard()
        # Listen for keypresses until escape is pressed
        keys = kb.getKeys()
        if '2' not in keys:
            # quit screen and exit
            event.waitKeys()
            core.quit()

        return


    def end_run(self):
        """
        finishes the run.
        converting the log of all responses to a dataframe and saving it
        showing a scoreboard with results from all the tasks
        showing a final text and waiting for key to close the stimuli screen
        """

        self.set_runfile_results(self.all_run_response, save = True)

        # present feedback from all tasks on screen
        self.show_scoreboard(self.task_obj_list, self.screen)

        # stop the eyetracker
        if self.eye_flag:
            self.stop_eyetracker()
            # get the edf file from Eyelink PC
            self.tk.receiveDataFile(self.tk_filename, self.tk_filename)

        # end experiment
        end_exper_text = f"End of run\n\nTake a break!"
        end_experiment = visual.TextStim(self.screen.window, text=end_exper_text, color=[-1, -1, -1])
        end_experiment.draw()
        self.screen.window.flip()

        # waits for a key press to end the experiment
        # event.waitKeys()
        # Make keyboard object
        kb = keyboard.Keyboard()
        # Listen for keypresses until escape is pressed
        keys = kb.getKeys()
        if 'space' in keys:
            # quit screen and exit
            self.screen.window.close()
            core.quit()


    def simulate_fmri(self, **kwargs):
        """
        mostly borrowed from fMRI_launchScan.py:
        https://github.com/psychopy/psychopy/blob/release/psychopy/demos/coder/experiment%20control/fMRI_launchScan.py)
        emulates sync (ttl) pulses. Emulation is to allow debugging script timing
        offline, without requiring a scanner (or a hardware sync pulse generator).
        """
        # settings for launchScan:
        MR_settings = {
            'TR': 1.000,       # duration (sec) per whole-brain volume
            'volumes': 330,    # number of whole-brain 3D volumes per scanning run
            'sync': 't',       # character to use as the sync timing event; assumed to come at start of a volume
            'skip': 5,         # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
            }
        MR_settings.update(**kwargs)

        # settings for the experiment
        self.set_info(debug = True, **kwargs)

        # win = visual.Window(fullscr=False, screen = 0)
        globalClock = core.Clock()

        # summary of run timing, for each key press:
        output = u'vol    onset key\n'
        for i in range(-1 * MR_settings['skip'], 0):
            output += u'%d prescan skip (no sync)\n' % i

        vol = launchScan(self.screen.window, settings = MR_settings,
                        globalClock=globalClock, mode = 'Test')

        duration = MR_settings['volumes'] * MR_settings['TR']
        run_loop = True
        # note: globalClock has been reset to 0.0 by launchScan()
        while run_loop:
            # allKeys = event.getKeys()
            # for key in allKeys:
            #     if key == MR_settings['sync']:
            #         onset = globalClock.getTime()
                    # do your experiment code at this point if you want it sync'd to the TR
            self.run()
            run_loop = False

            # else:
            #     # handle keys (many fiber-optic buttons become key-board key-presses)
            #     output += u"%3d  %7.3f %s\n" % (vol-1, globalClock.getTime(), str(key))
            #     if key == 'escape':
            #         output += u'user cancel, '
            #         break

            # waits for a key press to end the experiment
            # event.waitKeys()
            # quit screen and exit
            # win.close()
            # core.quit()
