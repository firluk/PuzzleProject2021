# reads and converts coordinates from segmentation module

from matplotlib import pyplot as plt

import numpy as np
# import imutils
import cv2

# Load image, resize smaller, HSV color threshold
image = cv2.imread('plots/full_downscale.png')
scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
image = cv2.resize(image, dim)
# image = imutils.resize(image, width=600)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
sensitivity = 5
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])
mask = cv2.inRange(hsv, lower_white, upper_white)
# Remove small noise on mask with morph open
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=3)
closing2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
# smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, smooth_kernel, iterations=3)
# result = cv2.bitwise_and(image, image, mask=opening)
# result[opening==0] = (255,255,255) # Optional for white background

ret,thresh = cv2.threshold(mask,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

long_contours = [c for c in contours if c.shape[0] > 150]

filled = cv2.drawContours(np.zeros_like(mask), long_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

plt.imshow(filled); plt.show(); plt.close();

(totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(filled)
# cv2.drawContours(mask, contours, -1, (127,127,127), 3)
# cv2.drawContours(mask, contours, 2, (127,127,127), 3)

# cv2.imshow('result', result)
cv2.imshow('mask', mask)
# cv2.imshow('closing', closing)
# cv2.imshow('opening', opening)
# cv2.imshow('closing2', closing2)
cv2.waitKey()


