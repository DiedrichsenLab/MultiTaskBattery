import random
import math
import numpy as np
import turtle

import psychopy.visual
import psychopy.event

win = psychopy.visual.Window(
    size=[400, 400],
    units="pix",
    fullscr=False
)

# n_dots = 6

# dot_xys = []

# # dot_x = random.uniform(0, 10)
# for dot in range(n_dots):
#     # dot_x = 2 - dot
#     # dot_y = math.sin(dot_x*math.pi)
#     # dot_y = 3 - (dot_x*2)/1.5
#     # dot_x = random.uniform(-200, 200)
#     # dot_y = random.uniform(-200, 200)
#     # dot_y = math.sin(2*dot_x*math.pi)
#     # dot_y =math.sin(dot_x ** 2)
#     n = 6
#     N = n*3+1
#     r = .7 # magnitude of the perturbation from the unit circle, 
#     angles = np.linspace(0,2*np.pi,N)
#     verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
#     dot_x = verts[:, 0]
#     dot_y = verts[:, 1]

    # dot_xys.append([dot_x, dot_y])

# dot_stim = psychopy.visual.ElementArrayStim(
#                                             win=win,
#                                             units="pix",
#                                             nElements=1,
#                                             elementTex=None,
#                                             elementMask="circle",
#                                             xys=[dot_xys[0]],
#                                             sizes=10
#                                         )
# dot_stim.draw()

# win.flip()

# n = 8
# N = n*3+1
# r = .7 # magnitude of the perturbation from the unit circle, 
# angles = np.linspace(0,2*np.pi,N)
# verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
# verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
# dot_x = verts[:, 0]
# dot_y = verts[:, 1]

# dot_xys = verts*100

# print(dot_xys.shape)


# from scipy.interpolate import interp1d

# # x = np.linspace(0, 100, num=11, endpoint=True)

# t = np.linspace(0, 10, num = 4, endpoint=True)
# x = 100*np.cos(t)
# y = 100*np.sin(t)

# # y = np.cos(-x**2/9.0)

# # f = interp1d(x, y)

# # f2 = interp1d(x, y, kind='cubic')

# circle_xys = [[x[i], y[i]] for i in range(len(x))] 
# b = np.random.choice(len(circle_xys), size = 2, replace=False)
# # print(b)
# # b1 = b

# dot_xys      = [circle_xys[i] for i in b]
# abs_y = np.abs(dot_xys[0][1]) + np.abs(dot_xys[1][1])
# abs_x = np.abs(dot_xys[0][0]) + np.abs(dot_xys[1][0])

# print(abs_y/abs_x)
# print(f"testing atan {math.degrees(math.atan(abs_y/abs_x))}")

# prob_angle = math.atan(abs_y/abs_y)
# print(prob_angle)
# #Drawclock_hand
# # clock_hand = psychopy.visual.Line(win=win, units = "pix", start = [0,0], end = [-80, 0], lineColor = 'black', lineWidth = 3, ori = 90)
# arrowVert = [(-1.4,0.5),(-1.4,-0.5),(0,-0.5),(0,-1.5),(1.5, 0),(0,1.5),(0,0.5)]
# arrow = psychopy.visual.ShapeStim(win, vertices=arrowVert, fillColor='black', size=5, lineColor='black', ori = 20)

# arrow.draw()
# win.flip()


# print(f"angles are {prob_angle}")

# circle = psychopy.visual.Circle(
#     win=win,
#     units="pix",
#     radius=100,
#     fillColor=[0, 0, 0],
#     lineColor=[1, 1, 1]
# )

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

x = np.array([ 0. ,  1.2,  1.9,  3.2,  4. ,  6.5])
y = np.array([ 0. ,  2.3,  3. ,  4.3,  2.9,  3.1])

t, c, k = interpolate.splrep(x, y, s=0, k=4)
print('''\
t: {}
c: {}
k: {}
'''.format(t, c, k))
N = 100
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)
yy = spline(xx)
print(yy.shape)
print(xx.shape)

dot_xys = np.vstack((xx*100, yy*100)).T

# print(a.shape)

win.flip()
for d in dot_xys:
    # print(d)
    for frameN in range(10):   # For exactly 200 frames
        print(d)
        # circle.draw()
        dot_stim = psychopy.visual.ElementArrayStim(
                                                        win=win,
                                                        units="pix",
                                                        nElements=1,
                                                        elementTex=None,
                                                        elementMask="circle",
                                                        xys=[d],
                                                        sizes=10
                                                    )

        dot_stim.draw()

        win.flip()

psychopy.event.waitKeys()

win.close()