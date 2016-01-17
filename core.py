import cv2

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
    threshRet, threshImg = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("Binary video", threshImg)

    key = cv2.waitKey(10)
    if key == ord('x'):
        cv2.destroyWindow("Cam video")
        videoCapture.release()
        break
