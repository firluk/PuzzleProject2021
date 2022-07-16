import os

import utils
import cv2 as cv
import numpy as np
import os
from piece import Piece, pieces_from_masks, masks_in_scale, image_in_scale
from facet import Facet
from puzzle_piece_detector.inference_callable import Inference

DEFAULT_WEIGHTS_PATH = './weights/mask_rcnn_puzzle.h5 '


def parse_args():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on Mask R-CNN to detect puzzle piece pieces.')
    parser.add_argument('--weights', required=True,
                        default=DEFAULT_WEIGHTS_PATH,
                        metavar="/path/to/mask_rcnn_puzzle.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=True,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    args = parser.parse_args()
    return args.weights, args.image


def main():
    # weights_path, image_path = parse_args()
    # inference = Inference(weights_path)
    # masks = inference.infer_masks(image_path)
    via_region_data_json_path, filename = 'dataset/12pieces/val/via_region_data.json', 'front_white.jpg'
    masks = utils.masks_from_via_region_data(via_region_data_json_path, filename)
    image_path = os.path.join(os.path.split(via_region_data_json_path)[0], filename)
    pieces = pieces_from_masks(masks, image_path)
    image = cv.imread(image_path)

    scale = 0.1
    masks = masks_in_scale(masks, scale)
    image = image_in_scale(image, scale)

    pieces = pieces_from_masks(masks, image)

    N = len(pieces)
    intersection_scores = np.zeros((N, N, 16))
    # malanohbis_distances = np.zeros((N, N, 16))
    # malanohbis_norm = 100
    for p1i, piece1 in enumerate(pieces):
        for p2i, piece2 in enumerate(pieces):
            if piece1 != piece2:
                for f1i, facet1 in enumerate(piece1.facets):
                    for f2i, facet2 in enumerate(piece2.facets):
                        intersection_scores[p1i, p2i, f1i * 4 + f2i] = Facet.intersection_score(facet1, facet2)
                        # malanohbis_distances[p1i, p2i, f1i * 4 + f2i] = Facet.malanohbis_distance(facet1, facet2, malanohbis_norm)

    print(intersection_scores)
    sorted_idx = np.flip(np.argsort(intersection_scores, axis=None))
    idx_mask = np.take_along_axis(intersection_scores, sorted_idx, axis=None)
    sorted_filtered_idx = sorted_idx[idx_mask]
    print(sorted_filtered_idx)


if __name__ == '__main__':
    main()
