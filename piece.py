from enum import Enum

import cv2 as cv
import numpy as np
from scipy.signal import find_peaks, savgol_filter

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
            rho, phi = (np.append(np.roll(rho, -argmin_rho), rho[argmin_rho]),
                        np.append(np.roll(phi, -argmin_rho), phi[argmin_rho]))

            smoothed_rho = savgol_filter(rho, window_length, 3)
            peaks, _ = find_peaks(smoothed_rho, height=0)

            from itertools import combinations

            def angle_between(p1, p2):
                ang1 = np.arctan2(*p1[::-1])
                ang2 = np.arctan2(*p2[::-1])
                return np.rad2deg((ang1 - ang2) % (2 * np.pi))

            peaks = np.mod(peaks + argmin_rho, x.shape)  # correct back shift
            peaks = np.sort(peaks)  # sort ascending

            min_comb_tuple = (np.inf, -1)
            min_comb_2nd_tuple = (np.inf, -1)
            for comb in (combinations(peaks, 4)):
                idx = np.array(comb)
                b = np.array((x[idx], y[idx]))
                a = np.roll(b, -1, axis=1)
                c = np.roll(b, 1, axis=1)
                ba = a - b
                bc = c - b
                angles = angle_between(ba, bc)
                angles_sum_of_squares = np.sum(np.power(angles - 90, 2))

                if angles_sum_of_squares < min_comb_tuple[0]:
                    min_comb_2nd_tuple = min_comb_tuple
                    min_comb_tuple = (angles_sum_of_squares, comb)
                elif angles_sum_of_squares < min_comb_2nd_tuple[0]:
                    min_comb_2nd_tuple = (angles_sum_of_squares, comb)

            if min_comb_2nd_tuple[0] < np.inf:

                comb = np.asarray(min_comb_tuple[1])
                idx = np.array(comb)
                b = np.array((x[idx], y[idx]))
                a = np.roll(b, -1, axis=1)
                norms = np.linalg.norm(a - b, axis=0)

                comb = np.asarray(min_comb_2nd_tuple[1])
                idx = np.array(comb)
                b = np.array((x[idx], y[idx]))
                a = np.roll(b, -1, axis=1)
                norms_2nd = np.linalg.norm(a - b, axis=0)

                min_max_norm_comb = min_comb_tuple[1] if np.max(norms) <= np.max(norms_2nd) else min_comb_2nd_tuple[1]

                idx = np.asarray(min_max_norm_comb)
            else:
                idx = np.asarray(min_comb_tuple[1])

            xy = x[idx], y[idx]

            _image = np.zeros_like(cropped_mask)
            for i in range(4):
                _image[xy[1][i], xy[0][i]] = 255

            return xy, idx

        corners, corners_indices = retrieve_corners()

        def create_facets():
            facets = []  # list of Facet objects

            contour_split = np.split(contour, corners_indices)
            contour_split[0] = np.vstack((contour_split[-1], contour_split[0]))

            for i in range(4):
                strip_coordinates = contour_split[i]
                facet = Facet(strip_coordinates, self, i)
                facets.append(facet)

            for i in range(4):
                prev_i = (i - 1) % 4
                next_i = (i + 1) % 4
                facets[i].prev_facet = facets[prev_i]
                facets[i].next_facet = facets[next_i]

            return facets

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
        self.type = resolve_type()
        self.piece_id = piece_id


def pieces_from_masks(masks, image):
    pieces = list()
    for i in range(masks.shape[-1]):
        pieces.append(Piece(masks[:, :, i], image, i))
    return pieces
