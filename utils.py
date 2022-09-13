import json
import os

import cv2 as cv
import numpy as np
import skimage.draw
import skimage.io
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


def image_in_scale(image, scale):
    height, width = image.shape[:2]
    scaled_height, scaled_width = int(height * scale), int(width * scale)
    scaled_image = cv.resize(image.astype(np.uint8), (scaled_width, scaled_height), interpolation=cv.INTER_LINEAR)
    return scaled_image


def masks_in_scale(masks, scale):
    if scale == 1:
        return masks
    masks_num = masks.shape[-1]
    height, width = masks.shape[:2]
    scaled_height, scaled_width = int(height * scale), int(width * scale)
    scaled_masks = np.zeros((scaled_height, scaled_width, masks_num))
    for i in range(masks_num):
        mask = masks[:, :, i]
        scaled_masks[:, :, i] = cv.resize(mask, (scaled_width, scaled_height), interpolation=cv.INTER_NEAREST)
    return scaled_masks.astype('uint8')


def image_with_contour_in_scale(image, contour, scale):
    height, width = image.shape[:2]
    scaled_height, scaled_width = int(height * scale), int(width * scale)
    scaled_image = np.zeros((scaled_height, scaled_width))
    unique_scaled_down = np.squeeze(np.unique(np.round(contour * scale).astype(np.int_), axis=0))

    def on_border(c, c_max):
        mask = np.equal(c, c_max)
        c[mask] = c_max - 1
        return c

    unique_scaled_down[:, 1] = on_border(unique_scaled_down[:, 1], scaled_height)
    unique_scaled_down[:, 0] = on_border(unique_scaled_down[:, 0], scaled_height)

    def coord_in_bounds(c, c_min, c_max):
        return np.logical_and(c_min <= c, c < c_max)

    x_filter = coord_in_bounds(unique_scaled_down[:, 1], 0, scaled_height)
    y_filter = coord_in_bounds(unique_scaled_down[:, 0], 0, scaled_width)
    xy_filter = np.logical_and(x_filter, y_filter)
    # scaled_image[unique_scaled_down[:, 1], unique_scaled_down[:, 0]] = 255
    scaled_image[unique_scaled_down[xy_filter, 1], unique_scaled_down[xy_filter, 0]] = 255
    return scaled_image


def infer_using_saturation_and_hue(image_path):
    image = cv.imread(image_path)
    # scale_percent = 60  # percent of original size
    # scale_percent = 100  # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    width = int(image.shape[1])
    # height = int(image.shape[0] * scale_percent / 100)
    height = int(image.shape[0])
    dim = (width, height)

    # resize image
    image = cv.resize(image, dim)
    # image = imutils.resize(image, width=600)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    sensitivity = 5
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Remove small noise on mask with morph open
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel, iterations=3)
    mask = opening

    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    long_contours = [c for c in contours if c.shape[0] > 150]

    filled = cv.drawContours(np.zeros_like(mask), long_contours, -1, (255, 255, 255), thickness=cv.FILLED)

    n_labels, piece_labels, _, _ = cv.connectedComponentsWithStats(filled)

    masks = np.zeros((image.shape[0], image.shape[1], n_labels - 1), dtype=np.uint8)

    for i in range(1, n_labels):
        mask = (piece_labels == i)
        roll_dist = int(width * 0.0052)
        mask2 = np.roll(mask, -roll_dist)
        mask = np.logical_and(mask, mask2)
        masks[mask, i - 1] = 255


    return masks


def print_sol(solution, pieces, name):
    # TODO: move to puzzle.py
    wh_max = np.max([[piece.cropped_image.shape[0] for piece in pieces], [piece.cropped_image.shape[1] for piece in pieces]])
    blank = np.zeros((wh_max, wh_max))
    for i, sol in enumerate(solution):
        fig = plt.figure()

        for j, cell in enumerate(sol.block.flatten()):
            ax = fig.add_subplot(sol.block.shape[0], sol.block.shape[1], j + 1)
            if cell is not None:
                if cell.facet_piece_ind == 1:
                    img = np.rot90(pieces[cell.piece_ind].cropped_image, -1)
                elif cell.facet_piece_ind == 2:
                    img = np.rot90(pieces[cell.piece_ind].cropped_image, 2)
                elif cell.facet_piece_ind == 3:
                    img = np.rot90(pieces[cell.piece_ind].cropped_image, 1)
                else:
                    img = pieces[cell.piece_ind].cropped_image
                plt.title(cell.piece_ind)
                plt.imshow(img)
                ax.axis('off')
            else:
                plt.imshow(blank)
                ax.axis('off')
        plt.savefig(f'plots/block_{name}_{i}.png')
        plt.close(fig)
