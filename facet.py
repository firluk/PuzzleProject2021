from enum import Enum

import cv2 as cv
import numpy as np


class Facet:
    class Type(Enum):
        FLAT = 1
        TAB = 2
        BLANK = 3

    def __init__(self, strip_coordinates, piece):
        """
        :param strip_coordinates: ndarray:(N,1,2) - np array of coordinates, relative to cropped image
        :param piece: Piece corresponding to Facet
        """

        def determine_type():
            epsilon = cv.arcLength(strip_coordinates, False) * 0.1
            approx = cv.approxPolyDP(strip_coordinates, epsilon, False)
            xy = np.squeeze(approx)
            if xy.shape[0] == 2:
                return Facet.Type.FLAT
            xy = align_along_x(xy)
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

        self.strip_coordinates = strip_coordinates
        self.strip_coordinates_approx = cv.approxPolyDP(strip_coordinates, 0, False)
        self.type = facet_type
        self.piece = piece

        # NOTE: moved to a function
        # self.strip_mask = None  # binary mask
        # self.strip_image = None  # pixel under mask
        # self.corners = None  # identifiers

        # self.metrics = None  # TODO: will be decided further down the dev cycle, stub for now

    def facet_mask(self):
        """
        Retrieve the binary mask of the strip, relative to supplied cropped mask
        :return: binary mask of the strip
        """
        shape = (self.piece.cropped_mask.shape[0], self.piece.cropped_mask.shape[1])
        strip_mask = cv.polylines(np.zeros(shape), [self.strip_coordinates], False, 255, 0)
        return strip_mask

    def facet_image(self):
        """
        Returns an image under the strip mask.
        :return: RGB image with only contour strip visible
        """
        strip_mask = self.facet_mask()
        strip_image = np.zeros_like(self.piece.cropped_image)
        strip_image[np.where(strip_mask)] = self.piece.cropped_image[np.where(strip_mask)]
        return strip_image

    def strip_image(self):
        return self.piece.cropped_image[self.strip_coordinates[:, 0, 1], self.strip_coordinates[:, 0, 0]]

    def corners(self):
        """
        :return: Corners of the facet - first and last coordinates
        """
        return self.strip_coordinates[0], self.strip_coordinates[-1]

    def intersection_score(self, other):
        if self.type is Facet.Type.FLAT or other.type is Facet.Type.FLAT:
            return np.NINF
        elif self.type == other.type:
            return np.NINF

        xy = (np.squeeze(self.strip_coordinates_approx))
        xy_ht_diff = xy[-1] - xy[0]
        length = np.sqrt(xy_ht_diff[0] ** 2 + xy_ht_diff[1] ** 2)

        other_xy = (np.squeeze(other.strip_coordinates_approx))
        other_xy_ht_diff = other_xy[-1] - other_xy[0]
        other_length = np.sqrt(other_xy_ht_diff[0] ** 2 + other_xy_ht_diff[1] ** 2)

        length_pad = 3
        shape_pad = 3
        max_length = np.ceil(max(length, other_length)).astype(np.int32) + length_pad
        shape = (2 * max_length + shape_pad, 2 * max_length + shape_pad)

        aligned_xy = align_along_x(xy)
        aligned_xy[:, 1] = aligned_xy[:, 1] + max_length
        aligned_xy = aligned_xy.astype(np.int32)

        aligned_facet_bitmap = cv.drawContours(np.zeros(shape), [aligned_xy], -1, 5, cv.FILLED)

        other_aligned_xy = align_along_x(other_xy)
        other_aligned_xy[:, 1] = other_aligned_xy[:, 1] * -1
        other_aligned_xy[:, 0] = np.max(other_aligned_xy[:, 0]) - other_aligned_xy[:, 0]
        other_aligned_xy[:, 1] = other_aligned_xy[:, 1] + max_length
        other_aligned_xy = other_aligned_xy.astype(np.int32)
        other_aligned_facet_bitmap = cv.drawContours(np.zeros(shape), [other_aligned_xy], -1, 5, cv.FILLED)

        intersection = np.logical_and(aligned_facet_bitmap, other_aligned_facet_bitmap)
        return np.sum(intersection)

    def malanohbis_distance(self, other, N):
        # facet_mask = self.facet_mask()
        # strip_image = self.strip_image()
        # one_dimensional_strip = strip_image[facet_mask]


        pass


def intersection_score(piece, other):
    return piece.intersection_score(other)


def malanohbis_distance(piece, other, N):
    return piece.malanohbis_distance(other, N)


def align_along_x(xy):
    theta = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
    c, s = np.cos(theta), np.sin(theta)
    rotation_mat = np.array([[c, -s], [s, c]])
    return (xy - xy[0, :]) @ rotation_mat
