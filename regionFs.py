import collections

import cv2
import numpy as np

import simpleFs as simF


def rotate_regions(contours, angles, centers, sizes):
    # Funkcija koja vrsi rotiranje regiona oko njihovih centralnih tacaka
    # contours: skup svih kontura [kontura1, kontura2, ..., konturaN]
    # angles:   skup svih uglova nagiba kontura [nagib1, nagib2, ..., nagibN]
    # centers:  skup svih centara minimalnih pravougaonika koji su opisani oko kontura [centar1, centar2, ..., centarN]
    # sizes:    skup parova (height,width) koji predstavljaju duzine stranica minimalnog
    #             pravougaonika koji je opisan oko konture [(h1,w1), (h2,w2), ...,(hN,wN)]
    # ret_val: rotirane konture
    ret_val = []
    for idx, contour in enumerate(contours):

        angle = angles[idx]
        cx, cy = centers[idx]
        height, width = sizes[idx]
        if width < height:
            angle += 90

        # Rotiranje svake tacke regiona oko centra rotacije
        alpha = np.pi / 2 - abs(np.radians(angle))
        region_points_rotated = np.ndarray((len(contour), 2), dtype=np.int16)
        for i, point in enumerate(contour):
            x = point[0]
            y = point[1]

            rx = np.sin(alpha) * (x - cx) - np.cos(alpha) * (y - cy) + cx
            ry = np.cos(alpha) * (x - cx) + np.sin(alpha) * (y - cy) + cy

            region_points_rotated[i] = [rx, ry]
        ret_val.append(region_points_rotated)

    return ret_val


def resize_region(region):
    resized = cv2.resize(region, (32, 32), interpolation=cv2.INTER_NEAREST)
    return resized


def select_contour(img):
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skin_ycrcb = cv2.inRange(img_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    blur = cv2.GaussianBlur(simF.erode(skin_ycrcb), (3, 3), 0)
    contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    cnt = None
    center = None
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area and area > 5000:
            center, size, angle = cv2.minAreaRect(c)
            # cv2.drawContours(img, contours, i, (255, 0, 0), 3)
            max_area = area
            cnt = c

    return cnt, center


def select_roi(image_orig, image_bin):
    contours_borders, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = []
    contour_angles = []
    contour_centers = []
    contour_sizes = []
    for contour in contours_borders:
        center, size, angle = cv2.minAreaRect(contour)
        xt, yt, h, w = cv2.boundingRect(contour)
        region_points = []
        for i in range(xt, xt + h):
            for j in range(yt, yt + w):
                dist = cv2.pointPolygonTest(contour, (i, j), False)
                if dist >= 0 and image_bin[j, i] == 255:  # da li se tacka nalazi unutar konture?
                    region_points.append([i, j])
        contour_centers.append(center)
        contour_angles.append(angle)
        contour_sizes.append(size)
        contours.append(region_points)

    # Postavljanje kontura u vertikalan polozaj
    contours = rotate_regions(contours, contour_angles, contour_centers, contour_sizes)

    regions_dict = {}
    for contour in contours:
        if not contours:
            return None, None, None

        if np.logical_not(any(contour[:, 0])):
            return None, None, None

        if np.logical_not(any(contour[:, 1])):
            return None, None, None

        min_x = min(contour[:, 0])
        max_x = max(contour[:, 0])
        min_y = min(contour[:, 1])
        max_y = max(contour[:, 1])

        region = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.int16)
        for point in contour:
            x = point[0]
            y = point[1]

            # Pretpostavimo da gornja leva tacka regiona ima apsolutne koordinate (100,100).
            # Ako uzmemo tacku sa koordinatama unutar regiona, recimo (105,105), nakon
            # prebacivanja u relativne koordinate tacka bi trebala imati koorinate (5,5) unutar
            # samog regiona.
            region[y - min_y, x - min_x] = 255

        regions_dict[min_x] = [resize_region(region), (min_x, min_y, max_x - min_x, max_y - min_y)]

    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    # region = np.array(regions_dict.values())

    sorted_rectangles = sorted_regions[:, 1]
    region_distances = [-sorted_rectangles[0][0] - sorted_rectangles[0][2]]
    for x, y, w, h in sorted_regions[1:-1, 1]:
        region_distances[-1] += x
        region_distances.append(-x - w)
    region_distances[-1] += sorted_rectangles[-1][0]

    return image_orig, sorted_regions[:, 0], sorted_rectangles, region_distances
    # return region[:, 0], sorted_rectangles  # , region_distances
