import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./img_dev/3.jpg")
plt.imshow(img)
plt.show()

resize = cv2.resize(img, (round(img.shape[1] / 10), round(img.shape[0] / 10)))

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()

thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh)
plt.show()

thresh = cv2.blur(thresh, ksize=(3, 3))
plt.imshow(thresh)
plt.show()
