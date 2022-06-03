import os

import utils
from piece import Piece, pieces_from_masks
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


if __name__ == '__main__':
    # weights_path, image_path = parse_args()
    # inference = Inference(weights_path)
    # masks = inference.infer_masks(image_path)

    via_region_data_json_path, filename = 'dataset/12pieces/val/via_region_data.json', 'front_white.jpg'
    masks = utils.masks_from_via_region_data(via_region_data_json_path, filename)
    image_path = os.path.join(os.path.split(via_region_data_json_path)[0], filename)

    pieces = pieces_from_masks(masks, image_path)
