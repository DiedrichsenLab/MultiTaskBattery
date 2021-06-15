# import libraries
import os
import pandas as pd
import numpy as np
import math
import glob
from experiment_code.screen import Screen


from psychopy import visual, core, event, gui # data, logging

screen = Screen()

positions = [(-9, -6), (0, -6), (9, -6),
                (-9, 0), (0, 0), (9, 0), 
                (-9, 6), (0, 6), (9, 6)]

# positions = [(-1, -6), (0, -6), (1, -6),
#                 (-1, 0), (0, 0), (1, 0), 
#                 (-1, 6), (0, 6), (1, 6)]
feedback_all = ['1', '2' ,'3', '4', '5', '6' ,'7', '8', '9']
for position, feedback in zip(positions, feedback_all):
    print(position)
    print(feedback)
    scoreboard = visual.TextStim(screen.window, text = feedback, color = [-1, -1, -1], pos = position, height = 0.8)
    scoreboard.draw()

screen.window.flip()