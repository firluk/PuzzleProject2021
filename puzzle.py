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
    image = cv.imread(image_path)

    scale = 0.1
    masks = masks_in_scale(masks, scale)
    image = image_in_scale(image, scale)

    pieces = pieces_from_masks(masks, image)

    N = len(pieces)
    O = 4

    iou = np.zeros((N, N, O, O))
    mgc = np.zeros((N, N, O, O))
    cmp = np.zeros((N, N, O, O))
    P = 25
    for p1i, p1idx in enumerate(pieces):
        for p2i, p2idx in enumerate(pieces):
            if p1i > p2i:
                for f1i, facet1 in enumerate(p1idx.facets):
                    for f2i, facet2 in enumerate(p2idx.facets):
                        iou[p1i, p2i, f1i, f2i] = Facet.iou(facet1, facet2)
                        mgc[p1i, p2i, f1i, f2i] = Facet.mgc(facet1, facet2, P)
                        cmp[p1i, p2i, f1i, f2i] = Facet.compatibility(facet1, facet2, P)

    # sort and filter in descending order
    edges_by_mgc = sort_and_filter(N, O, 0, mgc)
    edges_by_iou = sort_and_filter(N, O, 0, iou)
    edges_by_cmp = sort_and_filter(N, O, 0, cmp)

    for edge in edges_by_mgc:
        p1idx = edge[0]
        p2idx = edge[1]
        f1 = edge[2]
        f2 = edge[3]


def sort_and_filter(N, O, filter_val, weights):
    sort_idx = np.flip(np.argsort(weights, axis=None))  # descending order using flat
    mask = weights.flat > filter_val  # filter mask
    filtered_idx = sort_idx[mask[sort_idx]]  # cutting according to filter
    edges = np.transpose(np.vstack(np.unravel_index(filtered_idx, (N, N, O, O))))  # get indices
    return edges


if __name__ == '__main__':
    main()
