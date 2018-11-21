import cv2
import numpy as np
from Pic import Pictures

def nothing(x):
    pass

def Cue_Stick(filename):
    cv2.namedWindow('image')
    # Open the file
    img = cv2.imread('../Pictures/' + filename, cv2.IMREAD_COLOR)
    img = cv2.GaussianBlur(img, (5,5), 0)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def Display(hsv1, hsv2):
    cv2.createTrackbar('H', 'image', 10, 245, nothing)
    cv2.createTrackbar('S', 'image', 100, 255, nothing)
    cv2.createTrackbar('V', 'image', 100, 255, nothing)
    cv2.createTrackbar('S2', 'image', 100, 255, nothing)
    cv2.createTrackbar('V2', 'image', 100, 255, nothing)

    while (1):

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        h = cv2.getTrackbarPos('H', 'image')
        s = cv2.getTrackbarPos('S', 'image')
        v = cv2.getTrackbarPos('V', 'image')
        s2 = cv2.getTrackbarPos('S2', 'image')
        v2 = cv2.getTrackbarPos('V2', 'image')
        lower_stick = np.array([h - 10, s, v])
        upper_stick = np.array([h + 10, s2, v2])

        mask1 = cv2.inRange(hsv1, lower_stick, upper_stick)
        mask2 = cv2.inRange(hsv2, lower_stick, upper_stick)

        numpy_horizontal = np.hstack((mask1, mask2))

        cv2.imshow('image', numpy_horizontal)
    cv2.destroyAllWindows()


hsv1 = Cue_Stick(Pictures()[0])
hsv2 = Cue_Stick(Pictures()[1])

Display(hsv1, hsv2)