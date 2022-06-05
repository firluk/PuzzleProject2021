import cv2 as cv
import numpy as np
from scipy.signal import find_peaks


class Piece:
    def __init__(self, full_mask, image, piece_id):
        _, _, stats, centroids = cv.connectedComponentsWithStats(full_mask)
        left = stats[1, cv.CC_STAT_LEFT]
        top = stats[1, cv.CC_STAT_TOP]
        width = stats[1, cv.CC_STAT_WIDTH]
        height = stats[1, cv.CC_STAT_HEIGHT]
        self.cropped_mask = full_mask[top:top + height, left:left + width]  # cropped bitmap
        self.cropped_image = image[top:top + height, left:left + width]  # cropped rgb image
        self.cropped_image[self.cropped_mask == 0] = 0
        self.id = piece_id

        contours, _ = cv.findContours(self.cropped_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = contours[0]
        (center_x, center_y), _ = cv.minEnclosingCircle(contour)
        self.center = (center_x, center_y)

        def polar(_x, _y) -> tuple:
            """
            Turns cartesian coordinates to polar
            :param _x: x coordinate
            :param _y: y coordinate
            :return: (rho, phi): rho - length of the vector, phi - angle of the vector in radians
            """
            return np.hypot(_x, _y), np.degrees(np.arctan2(_y, _x))

        def cartesian(_rho, _phi):
            """
            Turns polar coordinates to cartesian
            :param _rho: length of the vector
            :param _phi:  angle of the vector in radians
            :return: (x,y) : x coordinate, y: y coordinate
            """
            return (_rho * np.cos(_phi)), (_rho * np.sin(_phi))

        def smooth(array, windows_size):
            """
            1D array smoothing, using convolution of windows size
            :param array:
            :param windows_size:
            :return:
            """
            out0 = np.convolve(array, np.ones(windows_size, dtype=int), 'valid') / windows_size
            r = np.arange(1, windows_size - 1, 2)
            start = np.cumsum(array[:windows_size - 1])[::2] / r
            stop = (np.cumsum(array[:-windows_size:-1])[::2] / r)[::-1]
            return np.concatenate((start, out0, stop))

        rho, phi = polar(contour[:, 0, 0] - center_x, contour[:, 0, 1] - center_y)

        rho = smooth(rho, 11)  # adjust the smoothing amount if necessary

        # compute number of "knobs"
        peaks = find_peaks(rho, height=0)
        n_knobs = len(find_peaks(rho, height=0)[0]) - 4
        # adjust those cases where the peak is at the borders
        if rho[-1] >= rho[-2] and rho[0] >= rho[1]:
            n_knobs += 1

        coordinates = None  # absolute coordinates in image
        self.corners = None  # [relative to center] / [absolute of cropped piece] list of (x,y) in CCW order
        self.facets = None  # list of Facet objects

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
