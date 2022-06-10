import cv2 as cv
import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_prominences

from facet import Facet

SAVGOL_WINDOW_RATIO = 0.04


class Piece:
    def __init__(self, full_mask, image, piece_id):
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

            window_length = int(cv.arcLength(contour, True) * 0.01 / 2)
            prominences = peak_prominences(smoothed_rho, peaks, window_length)[0]

            four_highest_prominence = np.argsort(prominences)[-4:]  # lowest 4
            four_highest_prominence = np.sort(four_highest_prominence)  # sort ascending
            four_highest_prominence = peaks[four_highest_prominence]  # contour indices
            four_highest_prominence = np.mod(four_highest_prominence + argmin_rho,
                                             x.shape)  # correct back shift
            four_highest_prominence = np.sort(four_highest_prominence)  # sort ascending

            four_four_highest_prominence_xy_coordinates = x[four_highest_prominence], y[four_highest_prominence]
            return four_four_highest_prominence_xy_coordinates, four_highest_prominence

        corners, corners_indices = retrieve_corners()

        def create_facets():
            lst = []  # list of Facet objects

            contour_split = np.split(contour, corners_indices)
            contour_split[0] = np.vstack((contour_split[-1], contour_split[0]))

            for i in range(4):
                strip_coordinates = contour_split[i]
                # # visual debugging
                from matplotlib import pyplot as plt
                plt.imshow(cv.polylines(np.zeros_like(cropped_mask), [strip_coordinates], False, 255, 3))
                plt.show()
                lst.append(Facet(strip_coordinates))

            return lst

        facets = create_facets()

        center = np.mean(corners)

        # fields
        self.id = piece_id
        self.left = left
        self.top = top
        self.cropped_mask = cropped_mask
        self.cropped_image = cropped_image
        self.contour = contour
        self.corners = corners
        self.center = center
        self.facets = facets

    def rotate_piece(self, angle):
        # TODO
        pass


def pieces_from_masks(masks, image_path):
    image = cv.imread(image_path)
    pieces = list()
    for i in range(masks.shape[-1]):
        pieces.append(Piece(masks[:, :, i], image, i))
    return pieces


# piece debugging
def main():
    from utils import masks_from_via_region_data
    import os

    via_region_data_json_path, filename = 'dataset/12pieces/val/via_region_data.json', 'front_white.jpg'
    masks = masks_from_via_region_data(via_region_data_json_path, filename)
    image_path = os.path.join(os.path.split(via_region_data_json_path)[0], filename)
    pieces = pieces_from_masks(masks, image_path)


if __name__ == '__main__':
    main()
