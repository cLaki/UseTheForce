import win32api

import cv2
import numpy as np
import pyautogui as pagui

import annFs
import regionFs as regF
import simpleFs as simF


def main():
    merge_images = False
    show_video_gray = False
    show_video_bin = True
    first_iter = True
    work = False
    left_down = False
    right_down = False
    scr_w = win32api.GetSystemMetrics(0)
    scr_h = win32api.GetSystemMetrics(1)
    prev_x1, prev_y1, prev_x2, prev_y2 = 0, 0, 0, 0
    imgNum = 30
    img_text = ''

    # Obucavanje mreze
    hand_state = annFs.prepare_training_data(merge_images)
    img_train = annFs.load_training_image()
    img_train_bin = simF.get_image_bin(simF.get_image_gray(img_train))
    # sel_img_train, shapes, rectangles, dots = regF.select_roi(img_train.copy(), img_train_bin)
    _, shapes, rectangles, _ = regF.select_roi(img_train.copy(), img_train_bin)
    inputs = simF.prepare_for_ann(shapes)
    outputs = simF.convert_output(hand_state)
    ann = annFs.create_ann()
    ann = annFs.train_ann(ann, inputs, outputs)  # MOZDA NEKI TEST DATA SET

    # Priprema i snimanje kamerom
    videoCapture = cv2.VideoCapture()
    videoCapture.open(0)

    # cursor_x = win32api.GetSystemMetrics(0) / 2
    # cursor_y = win32api.GetSystemMetrics(1) / 2

    while videoCapture.isOpened():
        _, img = videoCapture.read()
        cnt, center = regF.select_contour(img)
        drawing = np.zeros((np.size(img, 0), np.size(img, 1)), np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (255, 255, 255), 2)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # _, img_bin = cv2.threshold(blur, 70, 225, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if show_video_gray:
            disp_img = img.copy()
            cv2.putText(disp_img, img_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.imshow("Cam video", disp_img)

        # contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # max_area = 0
        # cntIdx = 0
        # for i in range(len(contours)):
        #     cnt = contours[i]
        #     area = cv2.contourArea(cnt)
        #     if area > max_area:
        #         max_area = area
        #         cntIdx = i
        #
        # cnt = contours[cntIdx]
        #
        # drawing = np.zeros((np.size(img, 0), np.size(img, 1)), np.uint8)
        # cv2.drawContours(drawing, [cnt], 0, (255, 255, 255), 2)

        if show_video_bin:
            disp_drw = drawing.copy()
            cv2.putText(disp_drw, img_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.imshow("Binary video", disp_drw)

        key = cv2.waitKey(5)
        if key == ord('s'):
            # cv2.imwrite('C:\Users\Laki\Documents\Faks\Soft Kompjuting\RukaGray_' + str(imgNum) + '.png', gray)
            cv2.imwrite('C:\Users\Laki\Documents\Faks\Soft Kompjuting\RukaDraw_' + str(imgNum) + '.png', drawing)
            cv2.imwrite('C:\Users\Laki\Documents\Faks\Soft Kompjuting\Ruka_' + str(imgNum) + '.png', img)
            imgNum += 1

        if key == ord('x'):
            cv2.destroyAllWindows()
            videoCapture.release()
            break

        if key == ord('v'):
            show_video_gray = not show_video_gray
            if not show_video_gray:
                cv2.destroyWindow('Cam video')

        if key == ord('b'):
            show_video_bin = not show_video_bin

        # Uzmi poziciju kursora
        cursor_x, cursor_y = win32api.GetCursorPos()

        if key == ord('p'):
            work = not work

        if work:
            prediction, rectangle = annFs.think_and_decide(ann, drawing)  # img_bin

            if np.logical_not(any(prediction[:, 0])):
                continue

            result = hand_state[np.argmax(prediction)]
            img_text = result

            # x1 = rectangle[0][2]
            # y1 = rectangle[0][3]
            # x2 = rectangle[0][0]
            # y2 = rectangle[0][1]
            x1 = -center[0]
            y1 = center[1]

            if first_iter:
                prev_x1 = x1
                prev_y1 = y1
                first_iter = False
                continue
            else:
                move_x = x1 - prev_x1
                move_y = y1 - prev_y1

                prev_x1 = x1
                prev_y1 = y1

                if move_x > 30:
                    cursor_x += move_x
                    if cursor_x < 0:
                        cursor_x = 0
                    elif cursor_x > scr_w:
                        cursor_x = scr_w
                if move_y > 30:
                    cursor_y += move_y
                    if cursor_y < 0:
                        cursor_y = 0
                    elif cursor_y > scr_h:
                        cursor_y = scr_h

                # pagui.moveTo(cursor_x, cursor_y, 0.1)
                pagui.moveRel(move_x, move_y)

                # Koristimo apsolutnu poziciju kursora, da ne bi smo sracunavali relativne vrednosti
                if result.startswith('clench'):
                    if not left_down:
                        left_down = True
                        pagui.mouseDown()
                elif result.startswith('thumb'):
                    if not right_down:
                        right_down = True
                        pagui.mouseDown(button='right')
                elif result.startswith('full'):
                    if left_down:
                        left_up(cursor_x, cursor_y)
                        left_down = False
                    if right_down:
                        right_up(cursor_x, cursor_y)
                        right_down = False

                        # surf = cv2.SURF(400)  # Speeded Up Robust Features
                        # kp, des = surf.detectAndCompute(drawing, None)


def left_up(c_x, c_y):
    pagui.mouseUp()
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_ABSOLUTE, c_x, c_y, 0, 0)


def right_up(c_x, c_y):
    pagui.mouseUp(button='right')
    # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP | win32con.MOUSEEVENTF_ABSOLUTE, c_x, c_y, 0, 0)


main()
