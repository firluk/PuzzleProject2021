from enum import Enum

import numpy as np
import cv2 as cv


class Facet:
    class Type(Enum):
        FLAT = 1
        TAB = 2
        BLANK = 3

    def __init__(self, strip_coordinates):
        def determine_type():
            epsilon = cv.arcLength(strip_coordinates, False) * 0.1
            approx = cv.approxPolyDP(strip_coordinates, epsilon, False)
            xy = np.squeeze(approx)
            if xy.shape[0] == 2:
                return Facet.Type.FLAT

            theta = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
            c, s = np.cos(theta), np.sin(theta)
            xy = xy @ np.array([[c, -s], [s, c]])
            xy -= xy[0, :]
            x_r, y_r = xy[:, 0], xy[:, 1]
            x_r_c, y_r_c = np.mean(x_r), np.mean(y_r)

            if y_r_c > 0:
                _facet_type = Facet.Type.TAB
            elif y_r_c < 0:
                _facet_type = Facet.Type.BLANK
            else:
                _facet_type = Facet.Type.FLAT
            return _facet_type

        facet_type = determine_type()

        self.strip_coordinates = strip_coordinates  # TODO: debate whether to squeeze or not to squeeze
        self.type = facet_type

        # NOTE: moved to a function
        # self.strip_mask = None  # binary mask
        # self.strip_image = None  # pixel under mask
        # self.corners = None  # identifiers

        # self.metrics = None  # TODO: will be decided further down the dev cycle, stub for now

    def strip_mask(self, cropped_mask):
        """
        Retrieve the binary mask of the strip, relative to supplied cropped mask
        :param cropped_mask
        :return: binary mask of the strip
        """
        if cropped_mask:
            strip_mask = cv.polylines(np.zeros_like(cropped_mask), [self.strip_coordinates], False, 255, 0)
        else:
            raise ValueError('Provide either cropped mask/image or shape to create a strip mask')

        return strip_mask

    def strip_image(self, cropped_image):
        """
        Returns an image under the strip mask.
        :param cropped_image: Image to copy the strip from
        :return: RGB image with only contour strip visible
        """
        strip_mask = self.strip_mask(cropped_image)
        strip_image = np.zeros_like(cropped_image)
        strip_image[strip_mask] = cropped_image[strip_mask]
        return strip_image

    def corners(self):
        """
        :return: Corners of the facet - first and last coordinates
        """
        return self.strip_coordinates[0], self.strip_coordinates[-1]
