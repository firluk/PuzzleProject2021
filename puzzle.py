"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


Derived from Balloon example by Waleed Abdulla as stated above.

------------------------------------------------------------

Usage:

    # Train a new model starting from pre-trained COCO weights
    python3 puzzle.py train --dataset=/path/to/puzzle/dataset --weights=coco


    # Apply color splash to an image
    python3 puzzle.py splash --weights=/path/to/weights/file.h5 --image=<path to file>


    # Outline puzzle polygons in the image
    python3 puzzle.py outline --weights=/path/to/weights/file.h5 --image=<path to file>


    # Retrieve puzzle polygons of the image to JSON file
    python3 puzzle.py retrieve --weights=/path/to/weights/file.h5 --image=<path to file>



"""

import datetime
import json
import os
import sys
from warnings import filterwarnings

import cv2 as cv
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class PuzzleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "puzzle"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + puzzle

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class PuzzleDataset(utils.Dataset):

    def load_puzzle(self, dataset_dir, subset):
        """Load a subset of the Puzzle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("puzzle", 1, "puzzle")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

                # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "puzzle",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a puzzle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "puzzle":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "puzzle":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PuzzleDataset()
    dataset_train.load_puzzle(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PuzzleDataset()
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


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def retrieve_masks(_model, image):
    """Detect objects in image, using given model.
    :param _model: Keras model
    :param image: RGB image [height, width, 3]
    :return: np boolean array [height, width, N], N - number of detected instances
    """
    # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))
    # Detect objects
    r = _model.detect([image], verbose=1)[0]
    return r['masks']


def detect_and_color_splash(_model):
    """
    Detect puzzle pieces, turn the image grayscale and splash color on detected masks.
    :param _model: Keras model
    """
    # Read image
    image = skimage.io.imread(args.image)
    masks = retrieve_masks(_model, image)
    # Splash color on image
    splash = color_splash(image, masks)
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)


def masks2polygons(masks):
    """
    Given a np binary masks array of ints, return a list of np arrays representing polygons
    :param masks: np uint8 array [height, width, N]
    :return: {list: N}{tuple: 1}('np.uint8' (C, 1, 2)), N - masks, C - coords
    """
    polys = []
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        polys.append(contours)
    return polys


def detect_and_outline(_model):
    """
    Detect and outline the borders of detected pieces. The file with outlined borders starts with 'outline_' and
    continued by timestamp.
    :param _model: Keras model
    """
    image = skimage.io.imread(args.image)
    masks = retrieve_masks(_model, image).astype('uint8')
    polys = masks2polygons(masks)
    for contours in polys:
        cv.drawContours(image, contours, -1, (255, 0, 0), 3)
    file_name = "outline_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, image)
    print("Saved to ", file_name)


def detect_and_retrieve_polys(_model):
    """
    Detect and save the detected pieces as list of coordinates in json format:
    [
        [[x1_1,y1_1],[x1_2,y1_2],...,[x1_C1,y1_C1]],
        [[x2_1,y2_1],[x2_2,y2_2],...,[x2_C2,y2_C2]],
        ...
        [[xN_1,yN_1],[xN_2,yN_2],...,[xN_CN,yN_CN]]
    ]
    N - detected pieces, Ci - coordinates of N-th piece
    :param _model: Keras model
    """
    image = skimage.io.imread(args.image)
    masks = retrieve_masks(_model, image).astype('uint8')
    polys = masks2polygons(masks)
    to_be_saved = list()
    for tup in polys:
        to_be_saved.append(np.squeeze(tup[0], axis=1).tolist())
    with open("polys.json", "w+") as json_file:
        json.dump(to_be_saved, json_file)


def detect_and_retrieve_masks(_model):
    """
    Detect pieces, save the masks as separate jpg files, save masks as numpy loadable masks.npy file and archive
    images and archive the resulting files to a single masks.zip
    :param _model: Keras model
    """
    image = skimage.io.imread(args.image)
    masks = retrieve_masks(_model, image).astype('uint8')
    if not os.path.exists('./masks/'):
        os.makedirs('./masks/')
    np.save('./masks/masks.npy', masks)
    for i in range(masks.shape[-1]):
        cv.imwrite(f"./masks/mask{i}.jpg", masks[:, :, i] * 255)
    import shutil
    shutil.make_archive('./masks', 'zip', './masks/')


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
                        help="'train', 'splash', 'outline' or 'retrieve'")
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
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash" \
            or args.command == "outline" \
            or args.command == "masks" \
            or args.command == "polys" \
            or args.command == "retrieve":
        assert args.image, \
            "Argument --image is required for inference"
    else:
        raise Exception("Please provide ")
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    # Configurations
    if args.command == "train":
        config = PuzzleConfig()
    else:
        class InferenceConfig(PuzzleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
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
    elif args.command == "splash":
        detect_and_color_splash(model)
    elif args.command == "outline":
        detect_and_outline(model)
    elif args.command == "polys":
        detect_and_retrieve_polys(model)
    elif args.command == "masks":
        detect_and_retrieve_masks(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'splash', 'outline' or 'retrieve'".format(args.command))
