# Create target file for different tasks
# @ Ladan Shahshahani Feb. 6 2021
import numpy as np
import pandas as pd
import math
# import experiment_code.constants as consts
import constants as consts

def visuospatial_order(nrun = 5, study_name = 'behavioral', 
                       dot_dur = 0.75, delay_dur = 0.5, 
                       prob_dur = 1, iti_dur = 0.5,
                       trial_dur = 6, task_dur = 30, 
                       hand = 'right', TR = 1, 
                       display_trial_feedback = True, num_trials = 1, 
                       circle_radius = 5, load = 6):
    """
    creates target file for the visuospatial_order task
    target file fields:
        trial_num (a column with no name - just the trial number starting from 0)
        load (number of dots)
        dot_dur (time duration when the dot remains on the screen and then disapears)
        delay_dur (duration of the delay)
        trial_type (True/False)
        prob_dots (the dots that are chosen for the prob number or coordinates?)
        prob_dur (duration when prob stayes on the screen)
        trial_dur (duration of the trial)
        display_trial_feedback (True/False)
        hand (left/right)
        iti_dur (duration of iti)
        start_time (time when the trial starts)
        end_time (time when the trial ends)
    """
    # path to save the target files
    path2task_target = consts.target_dir / study_name / 'visuospatial_order'
    consts.dircheck(path2task_target)

    # Version 1. Simple: Considers a circle with a certain radius and draws random dots from the circle

    # loop over runs and create target 
    for run in range(nrun):
        T = {} # dictionary that will be converted to a dataframe later

        n_trials = int(task_dur/(trial_dur + iti_dur)) # total number of trials

        # assign trial_types
        n_trials_T = int(n_trials/2)
        n_trials_F = n_trials - n_trials_T

        trials_True  = np.tile(True, n_trials_T)
        trials_False = np.tile(False, n_trials_F)

        trials_types = np.concatenate((trials_True, trials_False), axis = 0)

        # randomly shuffle trials
        np.random.shuffle(trials_types)

        # fill in fields
        T['trial_type']             = trials_types     
        T['trial_dur']              = [trial_dur for i in range(n_trials)]
        T['iti_dur']                = [iti_dur for i in range(n_trials)]
        T['delay_dur']              = [delay_dur for i in range(n_trials)]
        T['dot_dur']                = [dot_dur for i in range(n_trials)]
        T['prob_dur']               = [prob_dur for i in range(n_trials)]
        T['hand']                   = [hand for i in range(n_trials)]
        T['display_trial_feedback'] = [display_trial_feedback for i in range(n_trials)]
        T['start_time']             = [(trial_dur + iti_dur)*i for i in range(n_trials)]
        T['end_time']               = [(i+1)*trial_dur + i*iti_dur for i in range(n_trials)]
        T['circle_radius']          = [circle_radius for i in range(n_trials)]

        T['xys_stim'] = []
        T['xys_prob'] = []
        T['angle_prob'] = []
        for t in range(n_trials):

            ## Create the circle with a certain radius
            tt = np.linspace(0, 10000, num = 6, endpoint=True)
            ## Using the equations for the circle to create x and y
            x = circle_radius*np.cos(tt)
            y = circle_radius*np.sin(tt)
            
            circle_xys = [[x[i], y[i]] for i in range(len(x))]

            # randomly select from circle_xys
            dot_idx = np.random.choice(len(circle_xys), size = load, replace=False)

            dot_xys      = [circle_xys[i] for i in dot_idx]

            T['xys_stim'].append(dot_xys) 
            # randomly pick two of the dots for probe based on the trial type
            # get the trial_type for the current trial
            current_tt = T['trial_type'][t]

            # pick two dots
            rand_probs = np.random.choice(len(dot_xys), size = 2, replace = False)

            if ~ current_tt: # False trial
                # the trial is false so two wrong digits with wrong order can be generated
                # sort the indices in descending order to make sure that their order is flipped
                probs_idx = np.sort(rand_probs)[::-1]
            else: # True trial
                # sort the indices in ascending order to make sure that their order is conserved
                probs_idx = np.sort(rand_probs)
            
            probs_xys  = [dot_xys[i] for i in probs_idx]
            ## get the angle between the prob_dots
            abs_y = np.abs(probs_xys[0][1]) + np.abs(probs_xys[1][1])
            abs_x = np.abs(probs_xys[0][0]) + np.abs(probs_xys[1][0])
            prob_angle = math.degrees(math.atan(abs_y/abs_x))
            T['xys_prob'].append(probs_xys) 
            T['angle_prob'].append(prob_angle)

        df_tmp = pd.DataFrame(T)

        target_filename = path2task_target / f"visuospatial_order_{task_dur}sec_{run+1:02d}.csv"
        df_tmp.to_csv(target_filename)
    return

def sternber_order(nrun = 5, study_name = 'behavioral', 
                   digit_dur = 0.75, delay_dur = 0.5, 
                   prob_dur = 1, load_list = [4, 6], 
                   iti_dur = 0.5, trial_dur = 6, 
                   task_dur = 30, hand = 'right', 
                   display_trial_feedback = True, num_trials = 1, 
                   TR = 1):
    """
    creates target file for the sternberg_order task
    Args:
        nrun (int)                    - number of runs
        study_name (str)              - behavioral or fmri
        load? (int)                   - number of digits to show during encoding
        num_trials (int)              - number of repetitions per trial type
        iti_dur (float)               - iti duration in seconds
        delay_dur (float)             - duration of delay between encoding and prob in seconds
        digit_dur (float)             - duration when a digit stays on the screen in seconds
        prob_dur (float)              - duration when prob stays on the screen in seconds
        hand (str)                    - hand with which response needs to be made
        trial_dur_list (list)         - trial duration in seconds (first element: slow , second element: fast)
        TR (float)                    - TR in seconds
        task_dur (float)              - task duration in seconds
        display_trial_feedback (bool) - True: display feedback after trial, False: don't display feedback after trial
    target file fields:
        load (number of digits)
        stim (a string with digits including space between them)
        prob_stim (digits that will be shown during prob)
        condition_name (easy - medium - difficult)
        digit_dur (how long each digit stays on the screen)
        delay_dur (duration of the delay period)
        prob_dur (duration of the prob)
        trial_dur (trial duration)
        display_trial_feedback (True/False)
        trial_type (True/False)
        hand (left/right)
        iti_dur (iti duration)
        start_time (time when the trial starts)
        end_time (time when the trial ends)
    """
    # path to save the target files
    path2task_target = consts.target_dir / study_name / 'sternberg_order'
    consts.dircheck(path2task_target)

    # loop over runs and create target files
    for run in range(nrun):
        T = {} # dictionary that will be converted to a dataframe later

        n_trials = int(task_dur/(trial_dur + iti_dur)) # total number of trials

        # assign trial_types
        n_trials_T = int(n_trials/2)
        n_trials_F = n_trials - n_trials_T

        trials_True  = np.tile(True, n_trials_T)
        trials_False = np.tile(False, n_trials_F)

        trials_types = np.concatenate((trials_True, trials_False), axis = 0)

        # randomly shuffle trials
        np.random.shuffle(trials_types)

        # generate random numbers betweem 1 and 9 
        rand_nums = [np.random.choice(range(1, 10), size = 6, replace = False) for i in range(n_trials)]
        ## convert the random numbers to str and concatenate them
        stim_str = []
        for nums in rand_nums:
            rand_str = ""
            for x in nums: rand_str += str(x) + " "
            stim_str.append(rand_str)

        # fill in fields
        T['stim']                   = stim_str      
        T['trial_type']             = trials_types     
        T['trial_dur']              = [trial_dur for i in range(n_trials)]
        T['iti_dur']                = [iti_dur for i in range(n_trials)]
        T['delay_dur']              = [delay_dur for i in range(n_trials)]
        T['digit_dur']              = [digit_dur for i in range(n_trials)]
        T['prob_dur']               = [prob_dur for i in range(n_trials)]
        T['hand']                   = [hand for i in range(n_trials)]
        T['display_trial_feedback'] = [display_trial_feedback for i in range(n_trials)]
        T['start_time']             = [(trial_dur + iti_dur)*i for i in range(n_trials)]
        T['end_time']               = [(i+1)*trial_dur + i*iti_dur for i in range(n_trials)]


        # determine the prob stim for each trial based on trial type
        prob_stim = []
        for t in range(n_trials):
            # get the trial_type for the current trial
            current_tt = T['trial_type'][t]

            # get the current stims
            current_stim = T['stim'][t]
            current_stim_digits = current_stim.split()

            if ~ current_tt: # if it's False:
                # the trial is false so two wrong digits with wrong order can be generated
                ## generate two random numbers in the same range
                rand_probs = np.random.choice(range(1, 10), size = 2, replace = False)
                probs_str = [str(x) for x in rand_probs] # convert them to str
                ## are the numbers in the sequence? if they are get the indices
                prob_order = [current_stim_digits.index(x) for x in probs_str if x in current_stim_digits]

                ## if prob order is not empty, then the order has to be changed
                if len(prob_order)<=1: # at least one number is in the generated prob numbers
                    rand_str = ""
                    for dig in probs_str: rand_str += str(dig) + " " 
                    prob_stim.append(rand_str)
                else: # then the digit is not in the stim sequence
                    ## find the one that comes last and put it as the first digit in the prob
                    first_prob_digit = current_stim_digits[max(prob_order)]
                    last_prob_digit  = current_stim_digits[min(prob_order)]

                    # generate the prob stim and append it 
                    prob_stim.append(first_prob_digit + " " + last_prob_digit)

            else: # if it's True
                # pick two random digits from current stimulus
                probs_str = np.random.choice(current_stim_digits, 2, replace = False)
                # determine the order
                prob_order = [current_stim_digits.index(x) for x in probs_str]

                ## find the one that comes last and put it as the first digit in the prob
                first_prob_digit = current_stim_digits[min(prob_order)]
                last_prob_digit  = current_stim_digits[max(prob_order)]

                # generate the prob stim and append it 
                prob_stim.append(first_prob_digit + " " + last_prob_digit)
        T['prob_stim'] = prob_stim

        df_tmp = pd.DataFrame(T)

        target_filename = path2task_target / f"sternberg_order_{task_dur}sec_{run+1:02d}.csv"
        df_tmp.to_csv(target_filename)

    return

def finger_sequence(nrun = 5, study_name = 'behavioral', 
                    seq_length = 6, num_trials = 1, 
                    announce_time = 0, iti_dur = 0.5, 
                    trial_dur = 3.25, TR = 1, 
                    task_dur = 30, display_trial_feedback = True):
    """
    creates target file for the finger sequence task
    Args:
        nrun (int)                    - number of runs
        study_name (str)              - behavioral or fmri
        seq_length (int)              - length of sequence or number of digits
        num_trials (int)              - number of repetitions per trial type
        announce_time(float)          - time for announcing the trial
        iti_dur (float)               - iti duration in seconds
        trial_dur_list (list)         - trial duration in seconds (first element: slow , second element: fast)
        TR (float)                    - TR in seconds
        task_dur (float)              - task duration in seconds
        display_trial_feedback (bool) - True: display feedback after trial, False: don't display feedback after trial

    target file fields:
        trial_num (a column with no name - just the trial number starting from 0)
        sequence (a sequence of digits)
        condition_type (simple - complex)
        trial_type (None - no true of false response)
        display_trial_feedback (True/False)
        hand (left/right)
        iti_dur (duration of iti)
        start_time (time when the trial starts)
        end_time (time when the trial ends)
        trial_dur (trial duration)
    """

    # the sequences
    seq = {}
    ## EXperimental:
    seq['complex'] = ['1 3 2 4 3 2', '2 1 3 4 3 1', 
                      '3 2 4 1 4 2', '4 1 2 3 4 1']
    ## Control
    seq['simple'] = ['1 1 1 1 1 1', '2 2 2 2 2 2', 
                     '3 3 3 3 3 3', '4 4 4 4 4 4']

    # path to save the target files
    path2task_target = consts.target_dir / study_name / 'finger_sequence'
    consts.dircheck(path2task_target)

    # loop over runs and create target files
    for run in range(nrun):
        T = {} # dictionary that will be converted to dataframe
        
        n_trials = int(task_dur/(trial_dur + iti_dur)) # total number of trials
        n_trials_comp = int(n_trials/2)
        n_trials_simp = n_trials - n_trials_comp

        # hand assignment for complex trials
        ## left: 0, right: 1
        ## hands are assigned equally (half of trials are with left and half are with right hand)
        # n_trials_left_comp  = int(n_trials_comp/2)
        # n_trials_right_comp = n_trials_comp - n_trials_left_comp
        # left_trials_comp    = np.tile('left', (n_trials_left_comp, 1))
        # right_trials_comp   = np.tile('right', (n_trials_right_comp, 1))
        # trials_hands_comp   = np.vstack((left_trials_comp, right_trials_comp))

        # hand assignment for simple trials
        ## left: 0, right: 1
        ## hands are assigned equally (half of trials are with left and half are with right hand)
        # n_trials_left_simp  = int(n_trials_simp/2)
        # n_trials_right_simp = n_trials_simp - n_trials_left_simp
        # left_trials_simp    = np.tile('left', (n_trials_left_simp, 1))
        # right_trials_simp   = np.tile('right', (n_trials_right_simp, 1))
        # trials_hands_simp   = np.vstack((left_trials_simp, right_trials_simp))
        
        # shuffle hand assignments for complex and simple trials
        # np.random.shuffle(trials_hands_comp)
        # np.random.shuffle(trials_hands_simp)
        
        # shuffle sequences for complex and simple
        np.random.shuffle(seq['complex'])
        np.random.shuffle(seq['simple'])

        # fill in fields
        ## randomize order of complex and simple conditions across runs
        if run%2 == 0: # in even runs complex sequences come first
            T['condition_type'] = np.concatenate((np.tile('complex', n_trials_comp), np.tile('simple', n_trials_simp)), axis=0)
            T['sequence']       = np.concatenate((seq['complex'], seq['simple']), axis=0).T.flatten()
            # T['hand']           = np.concatenate((trials_hands_comp, trials_hands_simp), axis=0).T.flatten()
            T['hand']           = np.tile('right', n_trials).T.flatten()
        else: # in odd runs simple sequences come first
            T['condition_type'] = np.concatenate((np.tile('simple', n_trials_simp), np.tile('complex', n_trials_comp)), axis=0)
            T['sequence']       = np.concatenate((seq['simple'], seq['complex']), axis=0).T.flatten()
            # T['hand']           = np.concatenate((trials_hands_simp, trials_hands_comp), axis=0).T.flatten()
            T['hand']           = np.tile('left', n_trials).T.flatten()
        
        T['trial_dur']              = [trial_dur for i in range(n_trials)]
        T['iti_dur']                = [iti_dur for i in range(n_trials)]
        T['start_time']             = [(trial_dur + iti_dur)*i for i in range(n_trials)]
        T['end_time']               = [(i+1)*trial_dur + i*iti_dur for i in range(n_trials)]
        T['trial_type']             = ['None' for i in range(n_trials)]
        T['announce_time']          = [announce_time for i in range(n_trials)]
        T['display_trial_feedback'] = [display_trial_feedback for i in range(n_trials)]

        df_tmp = pd.DataFrame(T)

        target_filename = path2task_target / f"finger_sequence_{task_dur}sec_{run+1:02d}.csv"
        df_tmp.to_csv(target_filename)
    return

def language():
    """
    creates target file for the language? task
    target file fields:
        trial_num (a column with no name - just the trial number starting from 0)
    """
    pass

def flexion_extension(nrun = 5, study_name = 'behavioral', 
                      trial_dur = 14, iti_dur = 1, 
                      stim_dur = 1,
                      task_dur = 30, display_trial_feedback = False, 
                      TR = 1):
    """
    creates target file for the toe flexion extension task
    Args:
        nrun (int)                    - number of runs
        study_name (str)              - behavioral or fmri
        iti_dur (float)               - iti duration in seconds
        stim_dur (float)              - duration when the action (either flexion or extension) stays on the screen
        TR (float)                    - TR in seconds
        task_dur (float)              - task duration in seconds
        display_trial_feedback (bool) - True: display feedback after trial, False: don't display feedback after trial
    target file fields:
        trial_num (a column with no name - just the trial number starting from 0)
        stim (either flex or extend toes)
        trial_type (None - no true of false response)
        display_trial_feedback (True/False - always false because there are no real responses )
        foot (left/right)
        iti_dur (duration of iti)
        start_time (time when the trial starts)
        end_time (time when the trial ends)
        trial_dur (trial duration)
    """
    # path to save the target files
    path2task_target = consts.target_dir / study_name / 'flexion_extension'
    consts.dircheck(path2task_target)

    # loop over runs and create target files
    for run in range(nrun):
        T = {} # this will be converted to a dataframe and saved as target file

        n_trials       = int(task_dur/(trial_dur+iti_dur)) # total number of trials
        n_trials_left  = int(n_trials/2) # trials for left foot
        n_trials_right = n_trials - n_trials_left # trials for right foot

        # fill in fields
        T['stim']                   = ["flexion extension" for i in range(n_trials)]
        T['stim_dur']               = [stim_dur for i in range(n_trials)]
        T['trial_dur']              = [trial_dur for i in range(n_trials)]
        T['iti_dur']                = [iti_dur for i in range(n_trials)]
        T['start_time']             = [(trial_dur + iti_dur)*i for i in range(n_trials)]
        T['end_time']               = [(i+1)*trial_dur + i*iti_dur for i in range(n_trials)]
        T['trial_type']             = ['None' for i in range(n_trials)]
        T['display_trial_feedback'] = [display_trial_feedback for i in range(n_trials)]

        ## determine the foot
        # trials_left  = list(np.tile("left", n_trials_left).T.flatten())
        # trials_right = list(np.tile("right", n_trials_right).T.flatten())         

        ### foot assignment
        #### random makes it difficult!
        # np.random.shuffle(trials_foot)
        #### maybe make it so that every other trial is right foot?
        # trials_foot = trials_left + trials_right
        # trials_foot[::2] = trials_left 
        # trials_foot[1::2] = trials_right
        #### do nothing fancy and just concatenate them?
        # trials_foot = np.concatenate((trials_left, trials_right), axis = 0)

        # T['foot'] = trials_foot

        df_tmp = pd.DataFrame(T)
        
        target_filename = path2task_target / f"flexion_extension_{task_dur}sec_{run+1:02d}.csv"
        df_tmp.to_csv(target_filename)

    return

def visual_search():
    """
    creates target file for the visual_search task
    target file fields:
        trial_num (a column with no name - just the trial number starting from 0)
        stim ()
        condition_name (easy - medium - hard)
        trial_type (True/False)
        display_trial_feedback (True/False)
        hand (left/right)
        iti_dur (duration of iti)
        start_time (time when the trial starts)
        end_time (time when the trial ends)
        trial_dur (trial duration)
    For this task, another file is needed to determine position of the display
    display_pos file:
        trial (trial number)
        stim (?)
        xpos (x coordinate)
        ypos (y_coordinate)
        orientation (orientation of the stimuli)
    """
    pass

def theory_of_mind():
    """
    creates target file for the theory of mind task
    target file fields:
        trial_num (a column with no name - just the trial number starting from 0)
        story (stimulus that will be displayed)
        trial_type (True/False)
        trial_dur (trial duration)
        iti_dur (duration of iti)
        start_time (time when the trial starts)
        end_time (time when the trial ends)
    """
    pass

def rest(trial_dur = 10):
    """
    creates target file for the rest task
    Args:
        trial_dur (float) - duration of the trial in seconds(for rest it is set at 10 seconds as default)
    target file fields:
        There are no trials really. Just one start_time and end_time
    """
    # path to save the target files
    path2task_target = consts.target_dir / study_name / 'rest'
    consts.dircheck(path2task_target)

    T = {} # will be converted to a dataframe and saved as the target file

    T['hand']                   = 'None'
    T['iti_dur']                = 0
    T['star_time']              = 0
    T['stim']                   = 'fixation'
    T['trial_dur']              = trial_dur 
    T['display_trial_feedback'] = False

    df_tmp = pd.DataFrame(T)
        
    target_filename = path2task_target / f"rest_{trial_dur}sec_{run+1:02d}.csv"
    df_tmp.to_csv(target_filename)

def run_target():

    """
    create target files for the tasks
    """

    # finger_sequence()
    # sternber_order()
    # flexion_extension()
    visuospatial_order()

    return



run_target()