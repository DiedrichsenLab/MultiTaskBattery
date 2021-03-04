from psychopy import visual, core

# Setup stimulus
win = visual.Window([800, 800])

fixation = visual.GratingStim(win, tex=None, mask='gauss', sf=0, size=0.2,
    name='fixation', autoLog=False)



# fixation2 = visual.GratingStim(win, tex=None, mask='gauss', sf=0, size=0.02,
#     name='fixation', autoLog=False)

fixation.pos = (-0.1, -0.1)

# Let's draw a stimulus for 200 frames, drifting for frames 50:100
for frameN in range(10000):   # For exactly 200 frames
    fixation.pos += (0.005, -0.001)
    fixation.draw()
    win.flip()
