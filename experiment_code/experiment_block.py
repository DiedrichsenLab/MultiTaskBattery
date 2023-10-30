# Defines the Experiment as a class
# @ Ladan Shahshahani  - Maedbh King June 2021

# import libraries
import os
import pandas as pd
import numpy as np
import math
import glob
import sys

from psychopy import visual, core, event, gui # data, logging

import experiment_code.utils as ut
from experiment_code.task_blocks import task_map
from experiment_code.ttl import ttl
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
        self.__dict__.update(kwargs)

        # open screen and display fixation cross
        ### set the resolution of the subject screen here:
        self.stimuli_screen = Screen(screen_number=1)

        # connect to the eyetracker already
        if self.const.eye_tracker:
            import pylink as pl
            # create an Eyelink class
            ## the default ip address is 100.1.1.1.
            ## in the ethernet settings of the laptop,
            ## set the ip address of the EyeLink ethernet connection
            ## to 100.1.1.2 and the subnet mask to 255.255.255.0
            self.tk = pl.EyeLink('100.1.1.1')

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
            inputDlg.addField('Run File name (str):',initial = 'run_01.csv')      # run number (int)
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
            self.run_file_path = 'run_01.csv'
            self.wait_ttl = True

    def init_run(self):
        """initializing the run:
            making sure a directory is created for the behavioral results
            getting run file
            Initializes all the tasks for a run
        """

        # 1. get the run file info: creates self.run_info
        self.read_runfile()

        # 2. Initialize the all tasks that we need
        self.task_obj_list = [] # a list containing task objects in the run
        for t_num, task_info in self.run_info.iterrows():
            # create a task object for the current task and append it to the list
            TaskName = task_map[task_info.task_name]

            Task_obj  = TaskName(task_info,
                                 screen = self.stimuli_screen,
                                 const = self.const)

            self.task_obj_list.append(Task_obj)

        # 2. make subject folder in data/raw/<subj_id>
        subj_dir = self.const.data_dir / self.subj_id
        ut.dircheck(subj_dir) # making sure the directory is created!

        # 3. check if a file for the result of the run already exists
        # self.check_datafile()

        # 5. start the eyetracker if eyeflag = True
        if self.eye_tracker:
            self.start_eyetracker()


        # 6. initialize a list for responses
        self.all_run_response = []


    def run(self):
        """
        run a run of the experiment
        """
        print(f"running the experiment")
        self.start_timer()

        for t_num, task in enumerate(self.task_obj_list):
            print(f"{task.name}")

            # wait till it's time to start the task

            real_start_time = self.timer_info['global_clock'].getTime() - self.timer_info['t0']
            start_time = self.task_file_info['task_startTime'].values[0]
            print(f"\n{self.task_name}")
            print(f"real_start_time {real_start_time} - start_time {start_time}")
            self.wait_dur(self.task_file_info['task_startTime'].values[0])

            ## sending a message to the edf file specifying task name
            if self.eye_tracking:
                pl.sendMessageToFile(f"task_name: {task.name} start_track: {pl.currentUsec()} real start time {real_start_time} TR count {ttl.count}")

            # display the instruction text for the task. (instructions are task specific)
            task.display_instructions()

            # wait for a time period equal to instruction duration
            self.wait_dur(self.task_file_info['task_startTime'].values[0] + self.task_file_info['instruct_dur'].values[0])

            # show the stimuli for the task and collect responses
            task_response_df = task.run()

            # adding run information to response dataframe
            task_response_df['run_name'] = self.run_name
            task_response_df['run_iter'] = self.run_iter
            task_response_df['run_num']  = self.run_number
            # 8.7.3 get the response dataframe and save it
            fpath = consts.raw_dir / self.study_name/ 'raw' / self.subj_id / f"{self.study_name}_{self.subj_id}_{self.task_name}.csv"
            Task_obj.save_task_response(task_response_df, fpath)
            # save_response(task_response_df, self.study_name, self.subj_id, self.task_name)

            # log results
            # collect real_end_time for each task
            self.all_run_response.append({
                                            'real_start_time': real_start_time,
                                            'real_end_time': (self.timer_info['global_clock'].getTime() - self.timer_info['t0']),
                                            'run_name': self.run_name,
                                            'task_idx': t_num+1,
                                            'run_iter': self.run_iter,
                                            'run_num': self.run_info['run_num'],
            })

            # wait till it's time to end the task
            self.wait_dur(self.task_file_info['task_endTime'].values[0])

        # 3. ending the run
        self.end_run()


    def read_runfile(self):
        """Reads the run file and creates self.run_info
        """
        self.run_info = pd.read_csv(self.const.run_dir / self.run_filename)


    def start_timer(self):
        """
        starts the timer for the experiment (for behavioral study)
        Returns:
            timer_info(dict)    -   a dictionary with all the info for the timer. keys are:
            global_clock : the clock from psychopy?
            t0           : the start time
        """
        #initialize a dictionary with timer info
        self.timer_info = {}

        # wait for ttl pulse or not?
        if self.ttl_flag: # if true then wait

            ttl.reset()
            while ttl.count <= 0:
                # print out the text to the screen
                ttl_wait_text = f"Waiting for the scanner\n"
                ttl_wait_ = visual.TextStim(self.stimuli_screen.window, text=ttl_wait_text,
                                                pos=(0.0,0.0), color=self.stimuli_screen.window.rgb + 0.5, units='deg')
                ttl.check()

                ttl_wait_.draw()
                self.stimuli_screen.window.flip()

            # print(f"Received TTL pulse")
            # get the ttl clock
            self.timer_info['global_clock'] = ttl.clock
        else:
            self.timer_info['global_clock'] = core.Clock()

        self.timer_info['t0'] = self.timer_info['global_clock'].getTime()

        return

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

    def check_runfile_results(self):
        """
        Checks if a file for behavioral data of the current run already exists
        Args:
            experiment_info(dict)   -   a dictionary with all the info for the experiment (after user inputs info in the GUI)
        Returns:
            run_iter    -   how many times this run has been run:)
        """

        self.run_dir = self.const.data_dir / f"{self.study_name}_subj-{self.subj_id}_run-{self.run_number}.tsv"
        if os.path.isfile(self.run_dir):
            # load in run_file results if they exist
            self.run_file_results = pd.read_csv(self.run_dir)
            if len(self.run_file_results.query(f'run_name=="{self.run_name}"')) > 0:
                current_iter = self.run_file_results.query(f'run_name=="{self.run_name}"')['run_iter'].max() # how many times has this run_file been executed?
                self.run_iter = current_iter+1
            else:
                self.run_iter = 1
        else:
            self.run_iter = 1
            self.run_file_results = pd.DataFrame()

        return

    def set_runfile_results(self, all_run_response, save = True):
        """
        gets the behavioral results of the current run and returns a dataframe to be saved
        Args:
            all_run_response(list)    -   list of dictionaries with behavioral results of the current run
            save(bool)                -   if True, saves the run results. Default: True
        Returns:
            run_file_results(pandas dataframe)    -   a dataframe containing behavioral results
        """

        # save run results
        new_df = pd.concat([self.run_info['run_file'], pd.DataFrame.from_records(all_run_response)], axis=1)

        # check if a file with results for the current run already exists
        self.check_runfile_results()
        self.run_file_results = pd.concat([self.run_file_results, new_df], axis=0, sort=False)

        # save the run results if save is True
        if save:
            self.run_file_results.to_csv(self.run_dir, index=None, header=True)

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


    def wait_dur(self, wait_time):
        """
        waits for a certain amount of time specified in wait_time
        """
        while (self.timer_info['global_clock'].getTime() - self.timer_info['t0']) <= wait_time:
            if self.ttl_flag:
                ttl.check()
            else:
                pass

    def end_run(self):
        """
        finishes the run.
        converting the log of all responses to a dataframe and saving it
        showing a scoreboard with results from all the tasks
        showing a final text and waiting for key to close the stimuli screen
        """

        self.set_runfile_results(self.all_run_response, save = True)

        # present feedback from all tasks on screen
        self.show_scoreboard(self.task_obj_list, self.stimuli_screen)

        # stop the eyetracker
        if self.eye_flag:
            self.stop_eyetracker()
            # get the edf file from Eyelink PC
            self.tk.receiveDataFile(self.tk_filename, self.tk_filename)

        # end experiment
        end_exper_text = f"End of run\n\nTake a break!"
        end_experiment = visual.TextStim(self.stimuli_screen.window, text=end_exper_text, color=[-1, -1, -1])
        end_experiment.draw()
        self.stimuli_screen.window.flip()

        # waits for a key press to end the experiment
        # event.waitKeys()
        # Make keyboard object
        kb = keyboard.Keyboard()
        # Listen for keypresses until escape is pressed
        keys = kb.getKeys()
        if 'space' in keys:
            # quit screen and exit
            self.stimuli_screen.window.close()
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

        vol = launchScan(self.stimuli_screen.window, settings = MR_settings,
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
