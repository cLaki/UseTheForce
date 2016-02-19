import cv2
import numpy as np


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def get_image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def get_image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin


def invert(image):
    return 255 - image


# def display_image(image, color= False):
#     if color:
#         plt.imshow(image)
#     else:
#         plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def scale_to_range(image):
    # assert isinstance(image, np.ndarray)
    f = open('image_data.txt', 'w')
    f.write(str(image))
    f.close()

    scaled = image / 255
    return scaled


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def convert_output(outputs):
    return np.eye(len(outputs))


def winner(output):
    # return max(max(output))
    return max(enumerate(output), key=lambda x: x[1])[0]


def remove_noise(binary_image):
    ret_val = erode(dilate(erode(erode(binary_image))))
    return ret_val
