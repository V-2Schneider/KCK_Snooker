import cv2
import numpy as np
from Pic import Pictures

cap = cv2.VideoCapture('../Video/Snooker.mp4')
if (cap.isOpened()== False):
  print("Error opening video stream or file")

def nothing(x):
    pass

# warning! Cue_Stick is probably not needed anymore!
def Cue_Stick(filename):

    # Open the file
    img = cv2.imread('../Pictures/Small/' + filename, cv2.IMREAD_COLOR)
    img = cv2.GaussianBlur(img, (5,5), 0)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def Display():
    cv2.namedWindow('image')

    # Convert BGR to HSV


    cv2.createTrackbar('H', 'image', 14, 245, nothing)
    cv2.createTrackbar('S', 'image', 8, 255, nothing)
    cv2.createTrackbar('V', 'image', 88, 255, nothing)
    cv2.createTrackbar('S2', 'image', 255, 255, nothing)
    cv2.createTrackbar('V2', 'image', 255, 255, nothing)
    cv2.createTrackbar('linel', 'image', 180, 360, nothing)
    cv2.createTrackbar('lineg', 'image', 100, 200, nothing)

    while (1):

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h = cv2.getTrackbarPos('H', 'image')
        s = cv2.getTrackbarPos('S', 'image')
        v = cv2.getTrackbarPos('V', 'image')
        s2 = cv2.getTrackbarPos('S2', 'image')
        v2 = cv2.getTrackbarPos('V2', 'image')
        lower_stick = np.array([h - 5, s, v])
        upper_stick = np.array([h + 5, s2, v2])

        mask = cv2.inRange(hsv, lower_stick, upper_stick)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        edges = cv2.Canny(res, 200, 150, apertureSize=3)

        minLineLengthVal = 20
        maxLineGapVal = 3
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=minLineLengthVal, maxLineGap=maxLineGapVal)
        if (not (lines is None or len(lines) == 0)):
            for x in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[x]:
                    cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('image', res)
        cv2.imshow('frame', frame)
        #cv2.imshow('image', edges)
    cv2.destroyAllWindows()


Display()