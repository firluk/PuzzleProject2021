import cv2 as cv
import numpy as np
import skimage.io


class Piece:
    def __init__(self, full_mask, image, piece_id):
        # TODO: fields are taken from the Functions google sheet
        self.id = piece_id
        coordinates = None  # absolute coordinates in image
        self.corners = None  # [relative to center] / [absolute of cropped piece] list of (x,y) in CCW order
        self.facets = None  # list of Facet objects
        self.center = None  # is calculated by corners in 2 iterations of 4 point identification
        self.cropped_image = None  # cropped rgb image
        self.cropped_bitmap = None
        # TODO # maybe it is an outside metric associated with an ID of this piece
        # self.entropy = None  # is calculated according to image

        # TODO: fields are directly derived from the cv functions
        contours, _ = cv.findContours(full_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        _, _, stats, centroids = cv.connectedComponentsWithStats(full_mask)

        contour = contours[0]
        left = stats[1, cv.CC_STAT_LEFT]
        top = stats[1, cv.CC_STAT_TOP]
        width = stats[1, cv.CC_STAT_WIDTH]
        height = stats[1, cv.CC_STAT_HEIGHT]
        area = stats[1, cv.CC_STAT_AREA]
        centroid = centroids[1]

        # TODO: these functions are taken from Functions sheet
        def retrieveCorners():
            pass

        def createFacets():
            pass

        # TODO: not taken from Functions sheet, Piece.findCenter, Piece.findAngle, Piece.calcEntropy

    # TODO: Piece.rotatePiece from Functions sheet
    def rotatePiece(self, angle):
        pass

    # TODO: finish these


def pieces_from_masks(masks, image_path):
    image = skimage.io.imread(image_path)
    pieces = list()
    for i in masks.shape[-1]:
        pieces.append(Piece(masks[:, :, i], image, i))
    return pieces
