# test to get robot to run on raspbery pi startup
from __future__ import division
import time
from gpiozero import CamJamKitRobot
import cv2
import numpy as np
import RPi.GPIO as GPIO
cap = cv2.VideoCapture(0)


robot = CamJamKitRobot()

motorspeed = 0.3

motorforward = (motorspeed, motorspeed)

# robot.value = motorforward
# time.sleep(1)

def test_motor_mode():
    robot.forward()
    time.sleep(4)

    robot.backward()
    time.sleep(4)

    robot.left()
    time.sleep(4)

    robot.right()
    time.sleep(4)
    
def get_blue_score():
    return None

def hunt_mode():
    print 'Hunt mode activated'
    robot.forward()
    time.sleep(0.1)
    robot.stop()
    # time.sleep(2)

def destroy():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(25, GPIO.OUT)
    wheel = GPIO.PWM(25, 50)
    wheel.start(7.5)
    
def detect_blue_mode(threshold):
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lower_red = np.array([30,150,50])
        # upper_red = np.array([255,255,180])

        lower_blue = np.array([72, 61, 139])
        upper_blue = np.array([141, 238, 238])
         
        # mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        mask[mask > 0] = 1
        # cv2.imshow('frame',frame)
        blue_score = np.divide(np.sum(mask),float(frame.size) /3) * 100
        print blue_score
        if blue_score > threshold:
            print 'Begin hunt mode!'
            hunt_mode()


    
threshold = 10
# detect_blue_mode(threshold)
destroy()
