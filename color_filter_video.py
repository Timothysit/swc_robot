#!/usr/bin/env python2
from __future__ import division # make sure division is working on python2 in pi
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# some extra robot stuff (please move some other place)
hunt_threshold = 1000000000000000000
import time
# from gpiozero import CamJamKitRobot
# robot = CamJamKitRobot()

# file to save blue score (for live plotting / recording)
# from live_plot import animate
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib import style
style.use('fivethirtyeight')
file = open('blue_score.txt','w') 
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
frame_count = 0
def animate(i):
    graph_data = open('blue_score.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = [] 
    ys = [] 
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))

    ax1.clear()
    ax1.plot(xs, ys)

# ani = animation.FuncAnimation(fig, animate, interval = 10)
# plt.show(block=False) # this allows the code to continue

# def read_and_plot():
#    with open('blue_score.txt', 'r') as file:


while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    lower_blue = np.array([72, 61, 139])
    upper_blue = np.array([141, 238, 238])

    # mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # print(frame.shape)
    # print(mask.shape)
    # time.sleep(10)
    # compute blue score on the image
    mask[mask > 0] = 1
    # blue_score = np.sum(mask > 0) / frame.size 
    # blue_score = np.sum(mask) / (frame.size /3) * 100 # this works on XPS15 but not on pi; keep outputting 0 ...
    # may be to do with how divide works in different version of python
    # print(frame.size) # 921600 
    blue_score = np.divide(np.sum(mask),float(frame.size) /3) * 100 # the key to getting it to work in pi is to just float()
    # blue_score = np.sum(mask)
# frame has to be divided by 3 since its dimensions are
    # l x w x 3 (RGB values)
    # print(blue_score)
    # blue_score = np.round(blue_score, decimals = 2)
    print(str(blue_score))
    with open('blue_score.txt', 'a+') as file:
        file.write(str(frame_count) + ',' + str(blue_score)+ '\n')
    frame_count = frame_count + 1
    # print(blue_score)
    # current place hold for explore mode
    motorleft = (0.3, 0)
    # robot.value = motorleft
    # move robot if blue score > threshold
    if blue_score > hunt_threshold:
        print('Begin hunt!')
        robot.stop()
        time.sleep(1)
        robot.forward()
        time.sleep(1.5)
        robot.stop()


cv2.destroyAllWindows()
cap.release()

