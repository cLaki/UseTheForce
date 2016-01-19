import cv2
import numpy as np

cv2.namedWindow("Cam video", cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("Binary video", cv2.CV_WINDOW_AUTOSIZE)

videoCapture = cv2.VideoCapture()
videoCapture.open(0)
while videoCapture.isOpened():
    ret, img = videoCapture.read()
    assert isinstance(img, object)
    cv2.imshow("Cam video", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshRet, threshImg = cv2.threshold(blur, 70, 225, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("Binary video", threshImg)

    contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    cntIdx = 0
    cnt = contours[0]
    hull = None
    for i in range(len(contours)):
        cnt = contours[i]
        assert isinstance(cnt, object)
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            cntIdx = i

        cnt = contours[cntIdx]
        hull = cv2.convexHull(cnt)

    drawing = np.zeros((np.size(img, 0), np.size(img, 1)), np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (128, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (128, 0, 255), 2)
    cv2.imshow("Binary video", drawing)

    key = cv2.waitKey(10)
    if key == ord('x'):
        cv2.destroyWindow("Cam video")
        videoCapture.release()
        break
