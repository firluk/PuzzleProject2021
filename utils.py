import json
import os

import numpy as np
import skimage.draw
import skimage.io
import cv2 as cv
from matplotlib import pyplot as plt


def masks_from_via_region_data(via_region_data_json_path, filename):
    """
    Given a via_region_data.json file path and file name, returns bit mask of drawn polygon according to coordinates.
    :param via_region_data_json_path: path to VIA annotations file, containing coordinates of bit mask.
    :param filename: path to RGB image [height, width, 3]
    :return: np boolean array [height, width, N], N - number of detected instances
    """
    annotations = json.load(open(via_region_data_json_path))

    annotations = list(annotations.values())  # don't need the dict keys
    annotations = [a for a in annotations if a['regions']]
    annotation = [a for a in annotations if a['filename'] == filename][0]

    dataset_dir, _ = os.path.split(via_region_data_json_path)
    image_path = os.path.join(dataset_dir, filename)
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]
    masks = np.zeros((height, width, len(annotation['regions'])), dtype=np.uint8)
    for i, r in enumerate(annotation['regions']):
        shape_attributes = r['shape_attributes']
        all_points_x = shape_attributes['all_points_x']
        all_points_y = shape_attributes['all_points_y']
        rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
        masks[rr, cc, i] = 255

    return masks


def cart2pol(x, y):
    """
    Turns cartesian coordinates to polar
    :param x: x coordinate
    :param y: y coordinate
    :return: (rho, phi): rho - length of the vector, phi - angle of the vector in radians
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, theta):
    """
    Turns polat coordinate to cartesian
    :param rho: vector length
    :param theta: angle
    :return:
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated
