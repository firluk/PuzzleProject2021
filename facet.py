from enum import Enum

import cv2 as cv
import numpy as np


class Facet:
    class Type(Enum):
        FLAT = 1
        TAB = 2
        BLANK = 3

    def __init__(self, strip_coordinates, piece, facet_id, next_facet=None, prev_facet=None):
        """
        Constructor of Facet instance from given strip coordinates that are relative to the piece
        :param strip_coordinates: ndarray:(N,1,2) - np array of coordinates, relative to cropped image
        :param piece: Piece corresponding to Facet
        :param facet_id: identifier of the facet in the context of piece facets
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
            return facet_mask_outer(self)

        def facet_image():
            """
            Returns an image under the strip mask.
            :return: RGB image with only contour strip visible
            """
            strip_mask = self.facet_mask
            image = np.zeros_like(self.piece.cropped_image, dtype=np.uint8)
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

        self.facet_id = facet_id
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
        self.next_facet: Facet = next_facet
        self.prev_facet: Facet = prev_facet

    def facet_mask_custom_contour_size(self, contour_size=1):
        shape = (self.piece.cropped_mask.shape[0], self.piece.cropped_mask.shape[1])
        mask = cv.polylines(np.zeros(shape), [self.strip_coordinates], False, 255, contour_size).astype(np.bool_)
        return mask

    def calculate_aligned_bitmaps(self, other):
        blank, tab = self.assign_blank_and_tab(other)

        # epsilon = int(cv.arcLength(tab.strip_coordinates, False) * 0.1)
        # tab_coords_trim = tab.strip_coordinates[epsilon:-epsilon]
        # blank_coords_trim = blank.strip_coordinates[epsilon:-epsilon]
        # tab_xy = np.squeeze(cv.approxPolyDP(tab_coords_trim, 0, False))
        # blank_xy = np.squeeze(cv.approxPolyDP(blank_coords_trim, 0, False))

        tab_xy = (np.squeeze(tab.strip_coordinates_approx))
        blank_xy = (np.squeeze(blank.strip_coordinates_approx))

        tab_xy_ht_diff = tab_xy[-1] - tab_xy[0]
        tab_length = np.sqrt(tab_xy_ht_diff[0] ** 2 + tab_xy_ht_diff[1] ** 2)

        blank_xy_ht_diff = blank_xy[-1] - blank_xy[0]
        blank_length = np.sqrt(blank_xy_ht_diff[0] ** 2 + blank_xy_ht_diff[1] ** 2)

        length_pad = 3
        shape_pad = 3
        max_length = np.ceil(max(tab_length, blank_length)).astype(np.int32) + length_pad
        shape = (2 * max_length + shape_pad, 2 * max_length + shape_pad)
        color = 1

        aligned_tab_xy = align_along_x(tab_xy)
        aligned_tab_xy[:, 1] = aligned_tab_xy[:, 1] + max_length
        aligned_tab_xy = aligned_tab_xy.astype(np.int32)
        aligned_tab_xy = shift_to_top(aligned_tab_xy)
        aligned_tab_facet_bitmap = cv.fillPoly(np.zeros(shape), [aligned_tab_xy], color)
        # aligned_tab_facet_bitmap = cv.drawContours(np.zeros(shape), [aligned_tab_xy], -1, color, cv.FILLED)

        aligned_blank_xy = align_along_x(blank_xy)
        aligned_blank_xy[:, 1] = aligned_blank_xy[:, 1] * -1
        aligned_blank_xy[:, 0] = np.max(aligned_blank_xy[:, 0]) - aligned_blank_xy[:, 0]
        aligned_blank_xy[:, 1] = aligned_blank_xy[:, 1] + max_length
        aligned_blank_xy = aligned_blank_xy.astype(np.int32)
        aligned_blank_xy = shift_to_top(aligned_blank_xy)
        aligned_blank_facet_bitmap = cv.fillPoly(np.zeros(shape), [aligned_blank_xy], color)
        # aligned_blank_facet_bitmap = cv.drawContours(np.zeros(shape), [aligned_blank_xy], -1, color, cv.FILLED)
        return aligned_tab_facet_bitmap, aligned_blank_facet_bitmap

    def verify_facets_snappable(self, other) -> bool:
        if self.type is Facet.Type.FLAT or other.type is Facet.Type.FLAT:
            return False
        elif self.type == other.type:
            return False
        # TODO
        # elif (self.next_facet.type is Facet.Type.FLAT and other.prev_facet.type is not Facet.Type.FLAT) or \
        #         (self.prev_facet.type is Facet.Type.FLAT and other.next_facet.type is not Facet.Type.FLAT):
        #     return False
        else:
            return True

    def intersection_sum(self, other) -> int:
        if not self.verify_facets_snappable(other):
            return 0

        aligned_tab_facet_bitmap, aligned_blank_facet_bitmap = self.calculate_aligned_bitmaps(other)
        intersection = np.logical_and(aligned_tab_facet_bitmap, aligned_blank_facet_bitmap)
        return int(np.sum(intersection))

    def union_sum(self, other) -> int:
        if not self.verify_facets_snappable(other):
            return 0

        aligned_facet_bitmap, other_aligned_facet_bitmap = self.calculate_aligned_bitmaps(other)
        union = np.logical_or(aligned_facet_bitmap, other_aligned_facet_bitmap)
        return int(np.sum(union))

    def assign_blank_and_tab(self, other):
        if self.type == Facet.Type.BLANK and other.type == Facet.Type.TAB:
            return self, other
        elif other.type == Facet.Type.BLANK and self.type == Facet.Type.TAB:
            return other, self

    def iou(self, other) -> float:
        """
        IOU - Maximizes similarity
        :param other: Facet to calculate iou with
        :return: value between 0 and 1, 1 meaning identity of Facets
        """
        if not self.verify_facets_snappable(other):
            return 0

        intersection_sum = self.intersection_sum(other)
        union_sum = self.union_sum(other)
        return intersection_sum / union_sum

    def mgc(self, other, length_for_comparison) -> float:
        """
        Mahalanobis Gradient Compatibility
        :param other: Facet to calculate mahalanobis gradient compatibility with
        :param length_for_comparison: normalizing parameter, as facets can be of different sizes
        :return: mgc value
        """
        if not self.verify_facets_snappable(other):
            return 0

        if self.type == Facet.Type.TAB and other.type == Facet.Type.BLANK:
            tab, blank = self, other
        elif self.type == Facet.Type.BLANK and other.type == Facet.Type.TAB:
            blank, tab = self, other
        else:
            return 0

        dim = (1, length_for_comparison)

        tab_rear = cv.resize(tab.strip_image, dim).astype(np.int32)
        tab_rear2nd = cv.resize(tab.strip_image_2nd_level, dim).astype(np.int32)
        tab_gr = np.abs(tab_rear - tab_rear2nd)
        tab_gr_mean = np.mean(tab_gr, axis=0, keepdims=True)

        blank_rear = cv.resize(blank.strip_image, dim).astype(np.int32)
        blank_rear2nd = cv.resize(blank.strip_image_2nd_level, dim).astype(np.int32)
        blank_gr = np.flipud(np.abs(blank_rear - blank_rear2nd))
        blank_gr_mean = np.mean(blank_gr, axis=0, keepdims=True)

        tab_blank_gr = (np.abs(tab_gr_mean - blank_rear))

        tab_cov = np.cov(np.squeeze(tab_gr.T))
        blank_cov = np.cov(np.squeeze(blank_gr.T))

        tab_cov_inv = np.linalg.inv(tab_cov)
        blank_cov_inv = np.linalg.inv(blank_cov)

        gr1_diff = np.squeeze(abs(tab_blank_gr - tab_gr_mean))
        gr2_diff = np.squeeze(abs(tab_blank_gr - blank_gr_mean))

        mahalanobis_dist_tab_blank = \
            np.sqrt(np.sum(gr1_diff @ tab_cov_inv @ gr1_diff.T))

        mahalanobis_dist_blank_tab = \
            np.sqrt(np.sum(gr2_diff @ blank_cov_inv @ gr2_diff.T))

        return mahalanobis_dist_tab_blank + mahalanobis_dist_blank_tab

    def compatibility(self, other, P) -> float:
        """
        :param other: Facet to calculate scaled compatibility with
        :param P: normalizing parameter, as facets can be of different sizes
        :return:
        """
        return self.mgc(other, P) * (1 - self.iou(other))


def compatibility_func():
    return lambda mgc_val, iou_val: mgc_val * (1 - iou_val)


def align_along_x(xy):
    theta = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
    c, s = np.cos(theta), np.sin(theta)
    rotation_mat = np.array([[c, -s],
                             [s, c]])
    return (xy - xy[0, :]) @ rotation_mat


def shift_to_top(xy):
    return xy - (0, np.min(xy[:, 1]))


def facet_mask_outer(facet: Facet, as_bool=True):
    shape = (facet.piece.cropped_mask.shape[0], facet.piece.cropped_mask.shape[1])
    mask = cv.polylines(np.zeros(shape), [facet.strip_coordinates], False, 255, 1).astype(np.bool_)
    return mask if as_bool else mask.astype(np.uint8)
