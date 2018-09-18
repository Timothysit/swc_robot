# test to get robot to run on raspbery pi startup
from __future__ import division
import time
from gpiozero import CamJamKitRobot
import cv2
import numpy as np
import RPi.GPIO as GPIO

# things for color tracking
import imutils
from collections import deque

# cap = cv2.VideoCapture(0)


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

def explore_mode(spinDuration = 0.1, forwardDuration = 0.1, stopDuration = 0.25):
    # spinDuration = np.random.uniform(1, 2)
    print 'Explore mode...'
    spinDuration = 0.1
    print spinDuration 
    robot.left()
    time.sleep(spinDuration)

    robot.stop()
    time.sleep(stopDuration)
    #forwardDuration = np.random.uniform(2, 5)
    # print forwardDuration
    # time.sleep(forwardDuration)

def stuck_mode():
    return None

def hunt_mode(duration = 0.4):
    print 'Hunt mode activated'
    robot.forward()
    destroy(duration)
    time.sleep(duration)
    robot.stop()

def destroy(duration = 0.5):
    GPIO.setmode(GPIO.BCM)
    pin = 25
    GPIO.setup(pin, GPIO.OUT)
    freq = 200 # in Hz (originally 50)

    # while True:
    wheel = GPIO.PWM(pin, freq)
    dutyCycle = 1
    wheel.start(dutyCycle)
    time.sleep(duration)
    print 'Destroy!'

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


def temp_detect_blue_circle(numCircleThreshold = 1, showImage = True, huntMode = False):
    return None



def detect_blue_circle(numCircleThreshold = 1, showImage = True, huntMode = False):
    # define lower and upper boundaries of colors in HSV color space
    lower = {'blue':(97, 100, 117), 'red':(166, 84, 141)}
    upper = {'blue':(117,255,255), 'red':(186,255,255)}

    # define standard colors for circle around object (comment out)
    colors = {'blue':(255,0,0), 'red':(0,0,255)}
    camera =  cv2.VideoCapture(0)
    while True:
	# count for the number of circles 
	numCircle = 0
        # grab the current frame
        (grabbed, frame) = camera.read()
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=200) # 600 by default
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #for each color in dictionary check object in frame
        for key, value in upper.items():
            # construct a mask for the color from dictionary`1, then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(hsv, lower[key], upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
	    numCircle = len(cnts)
            if showImage == True:
                # only proceed if at least one contour was found
                if len(cnts) > 0:
		    print 'Circle score:'	
	            print len(cnts)
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # only proceed if the radius meets a minimum size. Correct this value for your obect's size
                    if radius > 0.5:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                        cv2.putText(frame,key + " ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
			# print 'Nothing happening here'
		    if huntMode == True:
			print 'Begin hunt mode!'
			hunt_mode()
		else:
			explore_mode()


        # show the frame to our screen
        # if showImage == True:
        #    cv2.imshow("Frame", frame)
	
	# print len(cnts)
        # activate hunt mode if there is more than one circle
        #if huntMode == True and numCircle > 0:
        #    print 'Begin hunt mode!'
        #    hunt_mode()


        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break



threshold = 10
# detect_blue_mode(threshold)
# detect_blue_circle(numCircleThreshold = 1, showImage = True, huntMode = True)
# destroy()

detect_blue_circle(numCircleThreshold = 1, showImage = True, huntMode = True)
explore_mode()

def main():
	explore_mode()
	






