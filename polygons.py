import cv2 as cv
import numpy as np

# we want to generate a bitmap and convert it polygon

masks = np.full((41, 31, 2), False)

masks[20:30, 15:25, 0] = True
masks[5:15, 5:10, 1] = True

masks = masks.astype('uint8')

masks0 = masks[:, :, 0]
masks1 = masks[:, :, 1]

contours0_simple, _ = cv.findContours(masks0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours1_simple, _ = cv.findContours(masks1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

contours0_none, _ = cv.findContours(masks0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours1_none, _ = cv.findContours(masks1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

im_gray = np.zeros_like(masks0)

masks_white = (np.sum(masks, -1) >= 1)
cv.drawContours(im_gray, contours0_simple, -1, 1, 3)
cv.drawContours(im_gray, contours1_none, -1, 1, 3)
overlap = np.logical_and(masks_white, im_gray)

cv.imwrite("_overlap.jpg", overlap.astype('uint8') * 255)
cv.imwrite("_masks_white.jpg", masks_white.astype('uint8') * 255)
cv.imwrite("_im_gray.jpg", im_gray.astype('uint8') * 255)
