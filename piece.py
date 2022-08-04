from enum import Enum

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_prominences

from facet import Facet
from utils import cart2pol

SAVGOL_WINDOW_RATIO = 0.04


class Piece:
    class Type(Enum):
        CORNER = 1
        SIDE = 2
        MIDDLE = 3

    def __init__(self, full_mask, image, piece_id):
        """
        Initializes Piece from boolean mask and reference RGB image
        :param full_mask:
        :param image:
        """
        _, _, stats, _ = cv.connectedComponentsWithStats(full_mask)
        left = stats[1, cv.CC_STAT_LEFT]
        top = stats[1, cv.CC_STAT_TOP]
        width = stats[1, cv.CC_STAT_WIDTH]
        height = stats[1, cv.CC_STAT_HEIGHT]

        cropped_mask = np.zeros((height, width), dtype='uint8')
        cropped_mask[:height, 0:width] \
            = full_mask[top:top + height, left:left + width]  # cropped bitmap

        cropped_image = np.zeros((height, width, 3), dtype='uint8')
        cropped_image[0:height, 0:width, :] \
            = image[top:top + height, left:left + width, :]  # cropped rgb image
        cropped_image[cropped_mask == 0] = 0

        contours, _ = cv.findContours(cropped_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = contours[0]

        def retrieve_corners():
            """
            Finds and retrieves the 4 corners of the piece
            :return: (corners, corners_indices) - corners - (x,y) coordinates relative to cropped image dimensions
                                                - corner_indices - corresponding to coordinates indices in contour
            """
            (center_x, center_y), _ = cv.minEnclosingCircle(contour)

            x, y = contour[:, 0, 0], contour[:, 0, 1]
            rho, phi = cart2pol(x - center_x, y - center_y)

            argmin_rho = np.argmin(rho)
            window_length = len(rho) * SAVGOL_WINDOW_RATIO
            window_length = np.ceil(window_length) // 2 * 2 + 1  # closest odd
            window_length = window_length.astype(int)
            rho, phi = np.append(np.roll(rho, -argmin_rho), rho[argmin_rho]), np.append(np.roll(phi, -argmin_rho),
                                                                                        phi[argmin_rho])


            smoothed_rho = savgol_filter(rho, window_length, 3)
            peaks, _ = find_peaks(smoothed_rho, height=0)

            window_length = max(int(cv.arcLength(contour, True) * 0.001), 5)
            prominences = peak_prominences(smoothed_rho, peaks, window_length)[0]

            four_highest_prominence = np.argsort(prominences)[-4:]  # lowest 4
            four_highest_prominence = np.sort(four_highest_prominence)  # sort ascending
            four_highest_prominence = peaks[four_highest_prominence]  # contour indices
            four_highest_prominence = np.mod(four_highest_prominence + argmin_rho,
                                             x.shape)  # correct back shift
            four_highest_prominence = np.sort(four_highest_prominence)  # sort ascending

            four_highest_prominence_xy_coordinates = x[four_highest_prominence], y[four_highest_prominence]
            #
            # def four_minimal_distance_dist():
            #     from itertools import combinations
            #     from bisect import insort
            #     combs = combinations(peaks, 4)
            #     norms = []
            #     for i, comb in enumerate(combs):
            #         pass
            #
            # four_minimal_distance_dist()


            # four_highest_prominence = np.sort(four_highest_prominence)  # sort ascending
            # four_highest_prominence = peaks[four_highest_prominence]  # contour indices
            # four_highest_prominence = np.mod(four_highest_prominence + argmin_rho,
            #                                  x.shape)  # correct back shift
            # four_highest_prominence = np.sort(four_highest_prominence)  # sort ascending
            #
            # four_highest_prominence_xy_coordinates = x[four_highest_prominence], y[four_highest_prominence]

            # plt.imshow(cropped_image)
            # plt.show()
            # plt.imshow(cropped_mask)
            # plt.show()
            # plt.plot(rho)
            # plt.show()
            # plt.plot(smoothed_rho)
            # plt.show()
            # plt.plot(prominences)
            # plt.show()

            image = np.zeros_like(cropped_mask)
            for i in range(4):
                image[four_highest_prominence_xy_coordinates[1][i], four_highest_prominence_xy_coordinates[0][i]] = 255

            plt.imshow(image)
            plt.show()

            return four_highest_prominence_xy_coordinates, four_highest_prominence

        corners, corners_indices = retrieve_corners()
        # plt.plot(corners[1], corners[0], marker='.', linestyle = 'None'); plt.show()

        def create_facets():
            lst = []  # list of Facet objects

            contour_split = np.split(contour, corners_indices)
            contour_split[0] = np.vstack((contour_split[-1], contour_split[0]))

            for i in range(4):
                strip_coordinates = contour_split[i]
                facet = Facet(strip_coordinates, self, i)
                lst.append(facet)

            return lst

        def resolve_type():
            flats = 0
            for facet in self.facets:
                if facet.type == Facet.Type.FLAT:
                    flats = flats + 1
            if flats == 2:
                return Piece.Type.CORNER
            elif flats == 1:
                return Piece.Type.SIDE
            elif flats == 0:
                return Piece.Type.MIDDLE
            else:
                raise ValueError(f"There are {flats} flats somehow - the piece is not valid")

        # fields
        self.left = left
        self.top = top
        self.cropped_mask = cropped_mask
        self.cropped_image = cropped_image
        self.contour = contour
        self.corners = corners
        self.center = np.mean(corners)
        self.facets = create_facets()
        img = np.zeros_like(cropped_image)
        facet_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]).astype(np.uint8)
        for fi in range(len(self.facets)):
            mask = self.facets[fi].facet_mask
            img[mask, :] = facet_colors[fi, :]
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        self.type = resolve_type()
        self.piece_id = piece_id


def image_in_scale(image, scale):
    height, width = image.shape[:2]
    scaled_height, scaled_width = int(height * scale), int(width * scale)
    scaled_image = cv.resize(image, (scaled_width, scaled_height), interpolation=cv.INTER_LINEAR)
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


def pieces_from_masks(masks, image):
    pieces = list()
    for i in range(masks.shape[-1]):
        pieces.append(Piece(masks[:, :, i], image, i))
    return pieces
