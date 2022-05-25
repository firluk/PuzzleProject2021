"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


Derived from Balloon example by Waleed Abdulla as stated above.

------------------------------------------------------------

Usage:


    # Apply color splash to an image
    python3 inference.py splash --weights=/path/to/weights/file.h5 --image=<path to file>


    # Outline puzzle piece polygons in the image
    python3 inference.py outline --weights=/path/to/weights/file.h5 --image=<path to file>


    # Retrieve puzzle piece polygons of the image to JSON file
    python3 inference.py polys --weights=/path/to/weights/file.h5 --image=<path to file>


    # Retrieve puzzle piece polygons of the image to JSON file
    python3 inference.py masks --weights=/path/to/weights/file.h5 --image=<path to file>


"""

import datetime
import json
import os
import sys
from warnings import filterwarnings

import cv2 as cv
import numpy as np
import skimage

from mrcnn.config import Config

# Filter some deprecation warnings for cleaner output, safe to delete this line
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")
# Logs path, required for mrcnn even for inference (-_-)
LOGS_PATH = os.path.join(ROOT_DIR, "logs")
# Output folder for inference
DEFAULT_OUTPUT_DIR = "./"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib


# Configurations
class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "puzzle piece_piece_detector_inference_config"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + puzzle piece

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


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
    print(f"Running on {args.image}")
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
    # file_name = f"{args.output}\\splash.png"
    file_name = os.path.join(args.output, "splash.png")
    skimage.io.imsave(file_name, splash)
    print(f"Saved to {file_name}")


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
    # file_name = f"{args.output}\\outline.png"
    file_name = os.path.join(args.output, "outline.png")
    skimage.io.imsave(file_name, image)
    print(f"Saved to {file_name}")


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
    # file_name = f"{args.output}\\polys.json"
    file_name = os.path.join(args.output, "polys.json")
    with open(file_name, "w+") as json_file:
        json.dump(to_be_saved, json_file)
    print(f"Saved polygons to {file_name}")


def detect_and_retrieve_masks(_model):
    """
    Detect pieces, save the masks as separate jpg files, save masks as numpy loadable masks.npy file and archive
    images and archive the resulting files to a single masks.zip
    :param _model: Keras model
    """
    image = skimage.io.imread(args.image)
    masks = retrieve_masks(_model, image).astype('uint8') * 255
    import shutil
    masks_dir = os.path.join(args.output, 'masks')
    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)
    os.makedirs(masks_dir)
    np.save(os.path.join(masks_dir, "masks.npy"), masks)
    for i in range(masks.shape[-1]):
        mask_i_path = os.path.join(masks_dir, f"mask{i}.png")
        print(mask_i_path)
        cv.imwrite(mask_i_path, masks[:, :, i])
    cv.imwrite(os.path.join(masks_dir, "masks_all.png"), np.sum(masks, -1, dtype='uint8'))
    shutil.make_archive(masks_dir, "zip", masks_dir)
    print(f"Saved to '{os.path.join(masks_dir, 'masks.zip')}")


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run inference on Mask R-CNN to detect puzzle piece pieces.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'splash', 'outline' or 'retrieve'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--output', required=False,
                        default=DEFAULT_OUTPUT_DIR,
                        metavar="/path/to/put/output/",
                        help='Directory where the inferences are put (default=./)')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "splash" \
            or args.command == "outline" \
            or args.command == "masks" \
            or args.command == "polys":
        assert args.image, \
            "Argument --image is required for inference"
    else:
        raise Exception("Please provide both command and image to run inference on")
    print("Weights: ", args.weights)

    args.output = os.path.abspath(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    config = InferenceConfig()
    config.display()
    # Create model
    if args.command:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=LOGS_PATH)
    # Select weights file to load
    weights_path = args.weights
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    # Train or evaluate
    if args.command == "splash":
        detect_and_color_splash(model)
    elif args.command == "outline":
        detect_and_outline(model)
    elif args.command == "polys":
        detect_and_retrieve_polys(model)
    elif args.command == "masks":
        detect_and_retrieve_masks(model)
    else:
        print(f"'{args.command}' is not recognized. "
              "Use 'polys', 'splash', 'outline' or 'masks'")
