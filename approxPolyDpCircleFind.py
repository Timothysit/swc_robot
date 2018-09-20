import cv2
import numpy as np
'''
Finds circles using contour --> approxPolyDp

References: 
http://layer0.authentise.com/detecting-circular-shapes-using-contours.html

The main tweak I added is to check of the contour is convex. 
I find that (at least for the case of balloons) this improved the performance 
of accepting blue balloon / rejecting any blue stuff significantly
'''
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    raw_image = frame
    # cv2.imshow('Original Image', raw_image)
    # cv2.waitKey(0)


    bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
    # cv2.imshow('Bilateral', bilateral_filtered_image)
    # cv2.waitKey(0)

    # Add colour filtering here 
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([97, 100, 117])
    upper_blue = np.array([117,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # erosion, then dilation, useful for removing noise
    kernel = np.ones((9,9),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # dilaton, then erosion, useful for closing small holes inside foreground object,
    # or small black points on the object
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('mask', mask)

    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    # cv2.imshow('Edge', edge_detected_image)
    # cv2.waitKey(0)

    # _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        # usually min = 8, max = 23
        circleMinVertex = 8 # if more than this many vertices, treat as circle
        circleMaxVertex = 30
        if ((len(approx) > circleMinVertex) & (len(approx) < circleMaxVertex) & (area > 30)
        & cv2.isContourConvex(approx)):
             contour_list.append(contour)
        # if cv2.isContourConvex(approx):
        #   contour_list.append(contour)

    cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',raw_image)
    if cv2.waitKey(1) == 1048689: #if q is pressed
            break