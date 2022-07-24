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

        def strip_image(coordinates):
            """
            Returns image under facet mask as 1DxRGB np.array
            :return: Image under facet mask as 1DxRGB np.array
            """
            coordinates_0 = coordinates[:, 0, 0]
            coordinates_1 = coordinates[:, 0, 1]
            image = self.piece.cropped_image
            ret = image[coordinates_1, coordinates_0]
            ret = np.expand_dims(ret, axis=1)
            return ret

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
        self.strip_image = strip_image(self.strip_coordinates)
        self.strip_coordinates_2nd_level = calculate_strip_coordinate_2nd_level()
        self.strip_image_2nd_level = strip_image(self.strip_coordinates_2nd_level)
        self.type = determine_type()

    def calculate_aligned_bitmaps(self, other):
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
        color = 1

        aligned_xy = align_along_x(xy)
        aligned_xy[:, 1] = aligned_xy[:, 1] + max_length
        aligned_xy = aligned_xy.astype(np.int32)
        aligned_facet_bitmap = cv.drawContours(np.zeros(shape), [aligned_xy], -1, color, cv.FILLED)

        other_aligned_xy = align_along_x(other_xy)
        other_aligned_xy[:, 1] = other_aligned_xy[:, 1] * -1
        other_aligned_xy[:, 0] = np.max(other_aligned_xy[:, 0]) - other_aligned_xy[:, 0]
        other_aligned_xy[:, 1] = other_aligned_xy[:, 1] + max_length
        other_aligned_xy = other_aligned_xy.astype(np.int32)
        other_aligned_facet_bitmap = cv.drawContours(np.zeros(shape), [other_aligned_xy], -1, color, cv.FILLED)

        return aligned_facet_bitmap, other_aligned_facet_bitmap

    def intersection_sum(self, other):
        if self.type is Facet.Type.FLAT or other.type is Facet.Type.FLAT:
            return 0
        elif self.type == other.type:
            return 0

        aligned_facet_bitmap, other_aligned_facet_bitmap = self.calculate_aligned_bitmaps(other)
        intersection = np.logical_and(aligned_facet_bitmap, other_aligned_facet_bitmap)
        return np.sum(intersection)

    def union_sum(self, other):
        if self.type is Facet.Type.FLAT or other.type is Facet.Type.FLAT:
            return 0
        elif self.type == other.type:
            return 0

        aligned_facet_bitmap, other_aligned_facet_bitmap = self.calculate_aligned_bitmaps(other)
        union = np.logical_or(aligned_facet_bitmap, other_aligned_facet_bitmap)
        return np.sum(union)

    def iou(self, other):
        """
        IOU - Maximizes similarity
        :param other: Facet to calculate iou with
        :return: value between 0 and 1, 1 meaning identity of Facets
        """
        if self.type is Facet.Type.FLAT or other.type is Facet.Type.FLAT:
            return 0
        elif self.type == other.type:
            return 0

        intersection_sum = self.intersection_sum(other)
        union_sum = self.union_sum(other)
        return intersection_sum / union_sum

    def mgc(self, other, P):
        """
        Mahalanobis Gradient Compatibility
        :param other: Facet to calculate mahalanobis gradient compatibility with
        :param P: normalizing parameter, as facets can be of different sizes
        :return: mgc value
        """
        if self.type is Facet.Type.FLAT or other.type is Facet.Type.FLAT:
            return np.inf
        elif self.type == other.type:
            return np.inf

        dim = (1, P)

        p1_rear = cv.resize(self.strip_image, dim).astype(np.int32)
        p1_rear2nd = cv.resize(self.strip_image_2nd_level, dim).astype(np.int32)
        gr_p1 = np.abs(p1_rear - p1_rear2nd)
        gr_p1_mean = np.mean(gr_p1, axis=0, keepdims=True)

        p2_rear = cv.resize(other.strip_image, dim).astype(np.int32)
        p2_rear2nd = cv.resize(other.strip_image_2nd_level, dim).astype(np.int32)
        gr_p2 = np.flipud(np.abs(p2_rear - p2_rear2nd))
        gr_p2_mean = np.mean(gr_p2, axis=0, keepdims=True)

        gr_p1p2 = (np.abs(p1_rear - p2_rear))

        p1_cov = np.cov(np.squeeze(gr_p1.T))
        p2_cov = np.cov(np.squeeze(gr_p2.T))

        p1_cov_inv = np.linalg.inv(p1_cov)
        p2_cov_inv = np.linalg.inv(p2_cov)

        gr1_diff = np.squeeze(abs(gr_p1p2 - gr_p1_mean))
        gr2_diff = np.squeeze(abs(gr_p1p2 - gr_p2_mean))

        mahalanobis_distp1p2 = \
            np.sqrt(np.sum(gr1_diff @ p1_cov_inv @ gr1_diff.T))

        mahalanobis_distp2p1 = \
            np.sqrt(np.sum(gr2_diff @ p2_cov_inv @ gr2_diff.T))

        return mahalanobis_distp1p2 + mahalanobis_distp2p1

    def compatibility(self, other, P):
        """
        :param other: Facet to calculate scaled compatibility with
        :param P: normalizing parameter, as facets can be of different sizes
        :return:
        """
        return self.mgc(other, P) * (1 - self.iou(other))


def align_along_x(xy):
    theta = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
    c, s = np.cos(theta), np.sin(theta)
    rotation_mat = np.array([[c, -s], [s, c]])
    return (xy - xy[0, :]) @ rotation_mat
