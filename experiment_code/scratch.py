# import psychopy.visual
# import psychopy.event

# win = psychopy.visual.Window(
#     size=[400, 400],
#     units="pix",
#     fullscr=False,
#     color=[1, 1, 1]
# )
# seq_colors = [[-1, -1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, -1]]
# my_seq = '1 2 3 2 1 4'
# seq_digits = my_seq.split(" ")
# x_pos = -40
# text_obj = []
# iseq = 0
# for d in seq_digits:
#     print(x_pos)
#     print(f"d {d}")
#     text_obj.append(psychopy.visual.TextStim(win = win, text = d, color=seq_colors[iseq], pos = [x_pos, 0]))
#     x_pos = x_pos + 20
#     iseq = iseq + 1

# for i in text_obj:
#     # print(f"draw {i}")
#     i.draw()
# win.flip()

# # text.draw()

# #win.flip()

# psychopy.event.waitKeys()

# win.close()
# import libraries
from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import time
import math
import glob

from psychopy import visual, core, event, gui # data, logging

import constants as consts
from screen import Screen
from ttl import ttl

win = visual.Window(
    size=[400, 400],
    units="pix",
    fullscr=False,
    color=[1, 1, 1]
)

rect = visual.Rect(
    win=win,
    units="pix",
    width=200,
    height=100,
    fillColor=[1, -1, -1],
    lineColor=[-1, -1, 1]
)

circle = visual.Circle(
    win=win,
    units="pix",
    radius=150,
    fillColor=[0, 0, 0],
    lineColor=[-1, -1, -1]
)

circle.draw()

rect.draw()

win.flip()

mouse = event.Mouse()
buttons, times = mouse.getPressed(getTime = True)


myclock = core.Clock()

while myclock.getTime() <= 10:
    if sum(mouse.getPressed()) and rect.contains(mouse):
        print("My shape was pressed")



