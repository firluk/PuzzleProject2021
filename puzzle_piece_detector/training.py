"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


Derived from Balloon example by Waleed Abdulla as stated above.

------------------------------------------------------------

Usage:

    # Train a new model starting from pre-trained COCO weights
    python3 puzzle.py training --dataset=/path/to/puzzle/dataset --weights=coco


    # Train a new model starting from pre-trained COCO weights
    python3 puzzle.py training --dataset=/path/to/puzzle/dataset --weights=mask_rcnn_coco.h5



"""

import os
import sys
from warnings import filterwarnings

from imgaug import augmenters as iaa

from config import PuzzlePieceDetectorConfig
from dataset import PuzzlePieceDataset

# Filter some deprecation warnings for cleaner output, safe to delete this line
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PuzzlePieceDataset()
    dataset_train.load_puzzle(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PuzzlePieceDataset()
    dataset_val.load_puzzle(args.dataset, "val")
    dataset_val.prepare()

    # Image Augmentation
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=50,
    #             layers='heads',
    #             augmentation=augmentation)

    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='all',
                augmentation=augmentation)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect puzzles.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/puzzle/dataset/",
                        help='Directory of the Puzzle dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        config = PuzzlePieceDetectorConfig()
        assert args.dataset, "Argument --dataset is required for training"
    else:
        raise Exception("Please provide a command")
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    # Configurations
    config.display()
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights.lower().endswith("mask_rcnn_coco.h5"):
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print(f"'{args.command}' is not recognized. "
              "Use 'train'")
