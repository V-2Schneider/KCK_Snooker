import cv2
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from PIL import Image, ImageEnhance
import time
from sympy import Symbol
x = Symbol('x')
buf=[]


cap = cv2.VideoCapture('../Video/Snooker.mp4')
if cap.isOpened() is False:
    print("Error opening video stream or file")


def nothing(x):
    pass


# warning! Cue_Stick is probably not needed anymore!
# nwm
def cueStick(file, contrast):
    # Open the file
    img = cv2.GaussianBlur(file, (5, 5), 0)
    # blur = cv2.bilateralFilter(file, 9, 75, 75)
    image = ImageEnhance.Contrast(Image.fromarray(img)).enhance(contrast)
    image = np.array(image)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv


def display():
    #cv2.namedWindow('image')
    cv2.namedWindow('frame')

    # Convert BGR to HSV
    #cv2.createTrackbar('H', 'image', 30, 245, nothing)
    #cv2.createTrackbar('S', 'image', 0, 255, nothing)
    #cv2.createTrackbar('V', 'image', 224, 255, nothing)
    #cv2.createTrackbar('S2', 'image', 255, 255, nothing)
    #cv2.createTrackbar('V2', 'image', 255, 255, nothing)
    #cv2.createTrackbar('linel', 'image', 0, 360, nothing)
    #cv2.createTrackbar('lineg', 'image', 0, 200, nothing)
    #cv2.createTrackbar('contrast', 'frame', 17, 250, nothing)

    while (1):

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        _, frame = cap.read()

        contrast = 17 #cv2.getTrackbarPos('contrast', 'frame')

        hsv = cueStick(frame, contrast)

        h = 30 #cv2.getTrackbarPos('H', 'image')
        s = 0 #cv2.getTrackbarPos('S', 'image')
        v = 224 #cv2.getTrackbarPos('V', 'image')
        s2 = 255 #cv2.getTrackbarPos('S2', 'image')
        v2 = 255 #cv2.getTrackbarPos('V2', 'image')
        lower_stick = np.array([h - 5, s, v])
        upper_stick = np.array([h + 5, s2, v2])

        mask = cv2.inRange(hsv, lower_stick, upper_stick)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        img_grey = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        th3 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        edges = cv2.Canny(res, 100, 150, apertureSize=3)
        edges = edges[0:189, 174:465]
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(edges, kernel, iterations=1)

        minLineLengthVal = 25
        maxLineGapVal = 2
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=minLineLengthVal, maxLineGap=maxLineGapVal)
        if (not (lines is None or len(lines) == 0)):
            buf = lines.copy()
        for x in range(0, len(buf)):
            for x1, y1, x2, y2 in buf[x]:
                points = [(x1, y1), (x2, y2)]
                x_coords, y_coords = zip(*points)
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords)[0]
                if not(m == 0):
                    y3 = 286
                    x3 = int((y3 - c)/m)
                    #print(x3)
                    if not((x3 > 1000) or (x3 < -1000)):
                        cv2.line(frame, (x1+174, y1), (x3+174, y3), (255, 255, 255), thickness=2)

        #cv2.imshow('image', res)
        cv2.imshow('frame', frame)
        #cv2.imshow('edges', edges)
        #cv2.imshow('hsv, after transformations', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        time.sleep(.02)
        # cv2.imshow('image', edges)
    cv2.destroyAllWindows()


display()
