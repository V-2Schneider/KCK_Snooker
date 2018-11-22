import cv2
import numpy as np
from Pic import Pictures

def nothing(x):
    pass

def Cue_Stick(filename):
    cv2.namedWindow('image')
    # Open the file
    img = cv2.imread('../Pictures/Small/' + filename, cv2.IMREAD_COLOR)
    img = cv2.GaussianBlur(img, (5,5), 0)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def Display(filename1, filename2):
    hsv1 = Cue_Stick(filename1)
    hsv2 = Cue_Stick(filename2)

    img1 = cv2.imread('../Pictures/Small/' + filename1, cv2.IMREAD_COLOR)
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)

    img2 = cv2.imread('../Pictures/Small/' + filename2, cv2.IMREAD_COLOR)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    cv2.createTrackbar('H', 'image', 10, 245, nothing)
    cv2.createTrackbar('S', 'image', 100, 255, nothing)
    cv2.createTrackbar('V', 'image', 100, 255, nothing)
    cv2.createTrackbar('S2', 'image', 255, 255, nothing)
    cv2.createTrackbar('V2', 'image', 255, 255, nothing)

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


        img3 = np.hstack((img1, img2))

        res = cv2.bitwise_and(img3, img3, mask=numpy_horizontal)
        edges = cv2.Canny(res, 200, 150, apertureSize=3)

        minLineLength = 10
        maxLineGap = 1
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        if (not (lines is None or len(lines) == 0)):
            for x in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[x]:
                    cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('image', res)
        #cv2.imshow('image', edges)
    cv2.destroyAllWindows()


Display(Pictures()[4], Pictures()[5])