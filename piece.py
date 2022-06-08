import cv2 as cv
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from facet import Facet

SAVGOL_WINDOW_RATIO = 0.04

PAD = 10


class Piece:
    def __init__(self, full_mask, image, piece_id):
        _, _, stats, _ = cv.connectedComponentsWithStats(full_mask)
        self.left = stats[1, cv.CC_STAT_LEFT]
        self.top = stats[1, cv.CC_STAT_TOP]
        self.width = stats[1, cv.CC_STAT_WIDTH]
        self.height = stats[1, cv.CC_STAT_HEIGHT]
        self.cropped_mask = np.zeros((self.height + PAD * 2, self.width + PAD * 2), dtype='uint8')
        self.cropped_mask[PAD:self.height + PAD, PAD:self.width + PAD] \
            = full_mask[self.top:self.top + self.height, self.left:self.left + self.width]  # cropped bitmap
        self.cropped_image = np.zeros((self.height + PAD * 2, self.width + PAD * 2), dtype='uint8')
        self.cropped_image[PAD:self.height + PAD, PAD:self.width + PAD] \
            = image[self.top:self.top + self.height, self.left:self.left + self.width]  # cropped rgb image
        self.cropped_image[self.cropped_mask == 0] = 0
        self.id = piece_id

        contours, _ = cv.findContours(self.cropped_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        self.contour = contours[0]
        (center_x, center_y), _ = cv.minEnclosingCircle(self.contour)

        # from matplotlib import pyplot as plt
        # plt.plot(self.cropped_mask)
        # plt.show()
        # contour_mask = cv.drawContours(np.zeros_like(self.cropped_mask), contours, -1, 255, 0)
        # plt.plot(contour_mask)
        # plt.show()

        def cart2pol(_x, _y):
            """
            Turns cartesian coordinates to polar
            :param _x: x coordinate
            :param _y: y coordinate
            :return: (rho, phi): rho - length of the vector, phi - angle of the vector in radians
            """
            _rho = np.sqrt(_x ** 2 + _y ** 2)
            _phi = np.arctan2(_y, _x)
            return _rho, _phi

        def pol2cart(_rho, _phi):
            """
            Turns polar coordinates to cartesian
            :param _rho: length of the vector
            :param _phi:  angle of the vector in radians
            :return: (x,y) : x coordinate, y: y coordinate
            """
            _x = _rho * np.cos(_phi)
            _y = _rho * np.sin(_phi)
            return _x, _y

        # # polygon approximation method for finding peaks
        # epsilon = 0.01 * cv.arcLength(contour, True)
        # approx = cv.approxPolyDP(contour, epsilon, True)
        # x, y = approx[:, 0, 0], approx[:, 0, 1]

        x, y = self.contour[:, 0, 0], self.contour[:, 0, 1]
        rho, phi = cart2pol(x - center_x, y - center_y)

        argmin_rho = np.argmin(rho)
        window_length = len(rho) * SAVGOL_WINDOW_RATIO
        window_length = np.ceil(window_length) // 2 * 2 + 1  # closest odd
        window_length = window_length.astype(int)
        rho, phi = np.append(np.roll(rho, -argmin_rho), rho[argmin_rho]), np.append(np.roll(phi, -argmin_rho),
                                                                                    phi[argmin_rho])

        smoothed_rho = savgol_filter(rho, window_length, 3)
        peaks = find_peaks(smoothed_rho, height=0)

        # from matplotlib import pyplot as plt
        # plt.plot(rho)
        # plt.show()
        # plt.plot(smoothed_rho)
        # plt.show()
        # plt.plot(rho)
        # for xc in peaks[0]:
        #     plt.axvline(x=xc)
        # plt.show()

        four_lowest_rho_peaks_indices = np.argsort(smoothed_rho[peaks[0]])[:4]  # lowest 4
        four_lowest_rho_peaks_indices = np.sort(four_lowest_rho_peaks_indices)  # sort ascending
        four_lowest_rho_peaks_indices = peaks[0][four_lowest_rho_peaks_indices]  # contour indices
        four_lowest_rho_peaks_indices = np.mod(four_lowest_rho_peaks_indices + argmin_rho, x.shape)  # correct back shift
        four_lowest_rho_peaks_indices = np.sort(four_lowest_rho_peaks_indices)  # sort ascending

        self.corners = x[four_lowest_rho_peaks_indices], y[four_lowest_rho_peaks_indices]
        self.center = np.mean(self.corners)
        self.facets = []  # list of Facet objects

        contour_split = np.split(self.contour, four_lowest_rho_peaks_indices)
        contour_split[0] = np.vstack(contour_split[-1], contour_split[0])

        for i in range(4):
            facet = Facet(self, )


        def retrieveCorners():
            pass

        def createFacets():
            pass

    # TODO: Piece.rotatePiece from Functions sheet
    def rotatePiece(self, angle):
        pass


def pieces_from_masks(masks, image_path):
    image = cv.imread(image_path)
    pieces = list()
    for i in masks.shape[-1]:
        pieces.append(Piece(masks[:, :, i], image, i))
    return pieces
