# reads and converts coordinates from segmentation module

import json, os
import numpy as np
import cv2
# import cropPiece_prototype as cpp
import math, time
import csv
import re

# # Parameters
# blur = 21
# canny_low = 15
# canny_high = 150
# min_area = 0.0005
# max_area = 0.95
# dilate_iter = 10
# erode_iter = 10
# mask_color = (0.0,0.0,0.0)
# # initialize video from the webcam
# video = cv2.VideoCapture(0)
# while True:
#     ret, frame = video.read()
#     if ret == True:
#         # Convert image to grayscale
#         image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Apply Canny Edge Dection
#         edges = cv2.Canny(image_gray, canny_low, canny_high)
#         edges = cv2.dilate(edges, None)
#         edges = cv2.erode(edges, None)
#         # get the contours and their areas
#         contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]
#         # Get the area of the image as a comparison
#         image_area = frame.shape[0] * frame.shape[1]
#
#         # calculate max and min areas in terms of pixels
#         max_area = max_area * image_area
#         min_area = min_area * image_area
#         # Set up mask with a matrix of 0's
#         mask = np.zeros(edges.shape, dtype = np.uint8)
#         # Go through and find relevant contours and apply to mask
#         for contour in contour_info:
#             # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
#             if contour[1] > min_area and contour[1] < max_area:
#                 # Add contour to mask
#                 mask = cv2.fillConvexPoly(mask, contour[0], (255))
#         # use dilate, erode, and blur to smooth out the mask
#         mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
#         mask = cv2.erode(mask, None, iterations=mask_erode_iter)
#         mask = cv2.GaussianBlur(mask, (blur, blur), 0)
#         # Ensures data types match up
#         mask_stack = mask_stack.astype('float32') / 255.0
#         frame = frame.astype('float32') / 255.0
#         # Blend the image and the mask
#         masked = (mask_stack * frame) + ((1-mask_stack) * mask_color)
#         masked = (masked * 255).astype('uint8')
#         cv2.imshow("Foreground", masked)
#         # Use the q button to quit the operation
#         if cv2.waitKey(60) & 0xff == ord('q'):
#             break
#     else:
#         break
# cv2.destroyAllWindows()
# video.release()

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
for i in range(len(contours)):
    if contours[i].shape[0] > 150:
        cv2.drawContours(mask, contours, i, (127, 127, 127), 3)
# cv2.drawContours(mask, contours, -1, (127,127,127), 3)
# cv2.drawContours(mask, contours, 2, (127,127,127), 3)

# cv2.imshow('result', result)
cv2.imshow('mask', mask)
# cv2.imshow('closing', closing)
# cv2.imshow('opening', opening)
# cv2.imshow('closing2', closing2)
cv2.waitKey()

# file1=open("proto/nominal.csv")
# file2=open("proto/temp.txt",'a')
#
# condition = False
#
# for line in file1:
#
#     if condition:
#         ind = [i for i, n in enumerate(line) if n == ','][3]
#         if line[ind+1] == '1':
#             file2.write(line.replace(line, line[:ind+1] + '0' + line[ind + 2:]))
#         else:
#             file2.write(line.replace(line, line[:ind + 1] + '1' + line[ind + 2:]))
#     else:
#         file2.write(line)
#     if line == "[Data]\n":
#         condition = True
#
# file1.close()
# file2.close()


# annotations = json.load(open('img_src/via_hamburg_json.json'))
# original image
# -1 loads as-is so if it will be 3 or 4 channel as the original
# image = cv2.imread('img_src/front_white.jpg', -1)
# imgAnnot = image.copy()

# # <editor-fold desc="parse coordinates">
#
# def angle_between_three_points(pointA, pointB, pointC):
#     x1x2s = math.pow((pointA[0] - pointB[0]), 2)
#     x1x3s = math.pow((pointA[0] - pointC[0]), 2)
#     x2x3s = math.pow((pointB[0] - pointC[0]), 2)
#
#     y1y2s = math.pow((pointA[1] - pointB[1]), 2)
#     y1y3s = math.pow((pointA[1] - pointC[1]), 2)
#     y2y3s = math.pow((pointB[1] - pointC[1]), 2)
#
#     cosine_angle = np.arccos(
#         (x1x2s + y1y2s + x2x3s + y2y3s - x1x3s - y1y3s) / (2 * math.sqrt(x1x2s + y1y2s) * math.sqrt(x2x3s + y2y3s)))
#
#     return np.degrees(cosine_angle)
#
#
# apx = list(annotations)
#
# temp = annotations[apx[4]]['regions']
# features = []
# pieces = []
# # corners = []
# for i, poly in enumerate(temp):
#     feature = []
#     for rowx, rowy in zip(poly['shape_attributes']['all_points_x'], poly['shape_attributes']['all_points_y']):
#         feature.append((rowx, rowy))
#     features.append(feature)
#     pieces.append(cpp.cropPiece(feature))
# # </editor-fold>
#
# # start = time.time()
# for feature in features:
#     for i in range(len(feature)):
#         if i<=len(feature)-3:
#             if angle_between_three_points(feature[i], feature[i+1], feature[i+2]) < 100 and angle_between_three_points(feature[i], feature[i + 1], feature[i + 2]) > 80:
#                 cv2.circle(imgAnnot,feature[i+1], 5, (0,0,255),-1)
#
#         elif i == len(feature)-2:
#             if angle_between_three_points(feature[i], feature[i+1], feature[0]) < 100 and angle_between_three_points(feature[i], feature[i + 1], feature[0]) > 80:
#                 cv2.circle(imgAnnot,feature[i+1], 5, (0,0,255),-1)
#         elif i == len(feature)-1:
#             if angle_between_three_points(feature[i], feature[0], feature[1]) < 100 and angle_between_three_points(feature[i], feature[0], feature[1]) > 80:
#                 cv2.circle(imgAnnot, feature[0], 5, (0, 0, 255), -1)
#
# # end = time.time()
# # print(end-start)
# cv2.imwrite('coors.png',imgAnnot)

# <editor-fold desc="contours">

# import cv2
# import numpy as np
#
# # Let's load a simple image with 3 black squares
# image = cv2.imread('crop_small.jpg')
# blur = cv2.GaussianBlur(image,(5,5),0)
# cv2.waitKey(0)
#
# # Grayscale
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Find Canny edges
# # Defining all the parameters
# t_lower = 20 # Lower Threshold
# t_upper = 80 # Upper threshold
# aperture_size = 3 # Aperture size
# L2Gradient = True # Boolean
#
# edged = cv2.Canny(blur, t_lower, t_upper, apertureSize= aperture_size, L2gradient= L2Gradient)
# cv2.waitKey(0)
#
# # Finding Contours
# # Use a copy of the image e.g. edged.copy()
# # since findContours alters the image
# contours, hierarchy = cv2.findContours(edged,
#                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)
#
# print("Number of Contours found = " + str(len(contours)))
#
# # Draw all contours
# # -1 signifies drawing all contours
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#
# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# </editor-fold>

# <editor-fold desc="rotation and padding">
# pad = 700
# padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (0,0,0))
#
# (h, w) = padded.shape[:2]
# center = (w / 2, h / 2)
# angle = 30
# scale = 1
# M = cv2.getRotationMatrix2D(center, angle, scale)
# rotated = cv2.warpAffine(padded, M, (w, h))
# cv2.imwrite('rotated.png',rotated)

# cornersRot = []
# angleRad = math.radians(angle)
# for corner in corners:
#     cornerRot = []
#     for x, y in list(corner):
#         # if (x * cos(angle) - y * sin(angle) >= 0) and (x * sin(angle) + y * cos(angle) >= 0):
#         cornerRot.append((int((x * cos(angleRad) + y * sin(angleRad))), int(-(x * sin(angleRad) - y * cos(angleRad)))))
#     cornersRot.append(cornerRot)
# imgAnnot3 = rotated.copy()
# for corner in cornersRot:
#     for i in corner:
#         cv2.circle(imgAnnot3, i, 5, (0, 0, 255), -1)
# cv2.imwrite('coorsRot.png', imgAnnot3)
# </editor-fold>

# <editor-fold desc="find angle between lines">
# https://stackoverflow.com/questions/28260962/calculating-angles-between-line-segments-python-with-math-atan2

# def dot(vA, vB):
#     return vA[0] * vB[0] + vA[1] * vB[1]
#
# def ang(lineA, lineB):
#     # Get nicer vector form
#     vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
#     vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
#     # Get dot prod
#     dot_prod = dot(vA, vB)
#     # Get magnitudes
#     magA = dot(vA, vA) ** 0.5
#     magB = dot(vB, vB) ** 0.5
#     # Get cosine value
#     cos_ = dot_prod / magA / magB
#     # Get angle in radians and then convert to degrees
#     angle = math.acos(dot_prod / magB / magA)
#     # Basically doing angle <- angle mod 360
#     ang_deg = math.degrees(angle) % 360
#     if ang_deg - 180 >= 0:
#         # As in if statement
#         return 360 - ang_deg
#     else:
#         return ang_deg
#
#
# lineA = ((174, 118), (325, 381))
# lineB = ((324, 117), (177, 383))
#
# c = np.concatenate((a,b),axis=1)

# </editor-fold>

# # <editor-fold desc="join blocks">
#
# # BlockA, BlockB: np.arrays
# # PieceA, PieceB: tuples
#
#
# def joinBlocks(BlockA, BlockB, PieceA, PieceB):
#
#     BlkA = BlockA
#     BlkB = BlockB
#
#     AL = PieceA[1]
#     BL = PieceB[1]
#     tmp = AL-BL
#     if tmp < 0:
#         BlkA = np.hstack((np.zeros((BlkA.shape[0],abs(tmp)), dtype=int),BlkA))
#     elif tmp > 0:
#         BlkB = np.hstack((np.zeros((BlkB.shape[0], abs(tmp)), dtype=int), BlkB))
#     AR = BlockA.shape[1] - PieceA[1] - 1
#     BR = BlockB.shape[1] - PieceB[1] - 1
#     tmp = AR-BR
#     if tmp < 0:
#         BlkA = np.hstack((BlkA,np.zeros((BlkA.shape[0],abs(tmp)), dtype=int)))
#     elif tmp > 0:
#         BlkB = np.hstack((BlkB,np.zeros((BlkB.shape[0], abs(tmp)), dtype=int)))
#     AT = PieceA[0]
#     BT = PieceB[0]
#     tmp = AT - BT
#     if tmp < 0:
#         BlkA = np.vstack((np.zeros((abs(tmp), BlkA.shape[1]), dtype=int), BlkA))
#     elif tmp > 0:
#         BlkB = np.vstack((np.zeros((abs(tmp), BlkB.shape[1]), dtype=int), BlkB))
#     AB = BlockA.shape[0] - PieceA[0] - 1
#     BB = BlockB.shape[0] - PieceB[0] - 1
#     tmp = AB - BB
#     if tmp < 0:
#         BlkA = np.vstack((BlkA, np.zeros((abs(tmp), BlkA.shape[1]), dtype=int)))
#     elif tmp > 0:
#         BlkB = np.vstack((BlkB, np.zeros((abs(tmp), BlkB.shape[1]), dtype=int)))
#
#     return [BlkA, BlkB, BlkA+BlkB]
#
# # </editor-fold>
#
# A = np.array(([0,0,0],[0,0,1],[1,1,1]))
# B = np.array(([1,0],[0,0],[0,0]))
# [AA, BB, joinedBlock] = joinBlocks(A,B,(2,2),(0,0))

