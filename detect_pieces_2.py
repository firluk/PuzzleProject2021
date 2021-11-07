import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# convenience method
def plt_show_no_axis(img, title=""):
    plt.axis('off')
    plt.title(title)
    plt.imshow(img)
    plt.show()


# parameters of the puzzle and image size
pieces = 60
scale = 1 / 10
desired_width = 2000

# read image, resize and convert to rgb
img = cv.imread("./img_dev/white_bg_crop_piece.jpg")
scale = desired_width / img.shape[1]
img = cv.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.blur(gray, (3, 3))

# prepare for contour finding by thresholding the image
thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
thresh = cv.blur(thresh, ksize=(5, 5))


# filter out edge one contours
def contour_inside_image_border(contour, image_shape) -> bool:
    bb = cv2.boundingRect(contour)
    return 0 < bb[0] and 0 < bb[1] and bb[2] < image_shape[1] and bb[2] < image_shape[1]


def acceptable_contour_area(contour, max_area):
    return cv2.contourArea(contour) < max_area


all_contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
indices = [i for i in range(len(all_contours)) if hierarchy[0][i][3] != -1]

