import numpy as np
import cv2 as cv


class Facet:

    def __init__(self, strip_coordinates, corners):
        # TODO
        self.strip_coordinates = strip_coordinates
        self.type = None  # flat, male, female  # TODO this after alignment and such
        # self.strip_mask = None  # binary mask
        # self.strip_image = None  # pixel under mask
        # self.corners = None  # identifiers
        # self.metrics = None  # TODO: will be decided further down the dev cycle, stub for now

    def strip_mask(self, cropped_mask=None, shape=None):
        if cropped_mask:
            strip_mask = cv.polylines(np.zeros_like(cropped_mask), [self.strip_coordinates], False, 255, 0)
        elif shape:
            strip_mask = cv.polylines(np.zeros(shape), [self.strip_coordinates], False, 255, 0)
        else:
            raise Exception('Provide either cropped mask/image or shape to create a strip mask')

        return strip_mask

    def strip_image(self, cropped_image=None, shape=None):
        strip_mask = self.strip_mask(cropped_image, shape)
        strip_image = np.array(cropped_image, copy=True)
        strip_mask[strip_mask == 0] = 0
        return strip_image
