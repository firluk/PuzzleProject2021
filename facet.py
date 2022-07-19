from enum import Enum

import cv2 as cv
import matplotlib.pyplot as plt
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
            """
            Determines the facet type based on centroid position
            :return: Facet.Type
            """
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

        def facet_mask():
            """
            Retrieve the binary mask of the strip, relative to supplied cropped mask
            :return: binary mask of the strip
            """
            shape = (self.piece.cropped_mask.shape[0], self.piece.cropped_mask.shape[1])
            mask = cv.polylines(np.zeros(shape), [self.strip_coordinates], False, 255, 1)
            return mask

        def facet_image():
            """
            Returns an image under the strip mask.
            :return: RGB image with only contour strip visible
            """
            strip_mask = self.facet_mask
            image = np.zeros_like(self.piece.cropped_image)
            image[np.where(strip_mask)] = self.piece.cropped_image[np.where(strip_mask)]
            return image

        def strip_image():
            """
            Returns image under facet mask as 1DxRGB np.array
            :return: Image under facet mask as 1DxRGB np.array
            """
            return self.piece.cropped_image[self.strip_coordinates[:, 0, 1], self.strip_coordinates[:, 0, 0]]

        def corners():
            """
            :return: Corners of the facet - first and last coordinates
            """
            return self.strip_coordinates[0], self.strip_coordinates[-1]

        def calculate_strip_coordinate_2nd_level():
            """
            Calculates the 2nd layer contour under the facet
            :return: 2nd layer contour
            """
            shape = (self.piece.cropped_mask.shape[0], self.piece.cropped_mask.shape[1])
            # polyline thickness == 2 for the second level layer
            strip_mask = cv.polylines(np.zeros(shape), [self.strip_coordinates], False, 255, 2)
            strip_mask_2nd_level = ((self.piece.cropped_mask - self.facet_mask) * strip_mask).astype(np.uint8)
            contour = cv.findContours(strip_mask_2nd_level, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0][0]
            head = strip_coordinates[0, :]
            tail = strip_coordinates[-1, :]
            head_idx = np.argmin(np.sum((np.squeeze(contour) - head) ** 2, axis=1))
            tail_idx = np.argmin(np.sum((np.squeeze(contour) - tail) ** 2, axis=1))
            if tail_idx > head_idx:
                return contour[head_idx:tail_idx, :]
            else:
                return contour[head_idx:tail_idx:-1, :]

        self.piece = piece
        self.strip_coordinates = strip_coordinates
        self.strip_coordinates_approx = cv.approxPolyDP(strip_coordinates, 0, False)
        self.corners = corners()  # identifiers
        self.facet_mask = facet_mask()  # binary mask
        self.facet_image = facet_image()  # pixel under mask
        self.strip_image = strip_image()
        self.strip_coordinates_2nd_level = calculate_strip_coordinate_2nd_level()
        self.type = determine_type()

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

    def mahalanobis_distance(self, other, N):
        one_dimensional_strip = self.strip_image[self.facet_mask]
        cv.resize(one_dimensional_strip, (1, N), cv.INTER_AREA)

        pass


def intersection_score(piece, other):
    return piece.intersection_score(other)


def mahalanobis_distance(piece, other, N):
    return piece.mahalanobis_distance(other, N)


def align_along_x(xy):
    theta = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
    c, s = np.cos(theta), np.sin(theta)
    rotation_mat = np.array([[c, -s], [s, c]])
    return (xy - xy[0, :]) @ rotation_mat
