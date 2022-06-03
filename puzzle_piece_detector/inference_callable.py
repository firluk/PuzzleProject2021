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

import os
import sys
from warnings import filterwarnings

import cv2 as cv
import skimage

from puzzle_piece_detector.inference_config import InferenceConfig

# Filter some deprecation warnings for cleaner output, safe to delete this line
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")
# Logs path, required for mrcnn even for inference (-_-)
LOGS_PATH = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib


class Inference:

    def __init__(self, weights_path):
        self.model = None

        def load_model(weights_path):
            assert weights_path
            # Create config
            config = InferenceConfig()
            config.display()

            # Create model
            self.model = modellib.MaskRCNN(mode="inference", config=config,
                                           model_dir=LOGS_PATH)

            # Load weights
            print("Loading weights ", weights_path)
            self.model.load_weights(weights_path, by_name=True)

        load_model(weights_path)

    def infer_masks(self, image_path):
        """
        Detect objects in image, using given model, returns binary bitmap masks.
        :param image_path: path to RGB image [height, width, 3]
        :return: np boolean array [height, width, N], N - number of detected instances
        """
        # Run model detection and generate the color splash effect
        print(f"Running on {image_path}")
        image = skimage.io.imread(image_path)
        # Detect objects
        r = self.model.detect([image], verbose=1)[0]
        return r['masks']

    def infer_polys(self, image_path):
        """
        Detect objects in image, using given model, returns a list of polygons in form of coordinates (x,y).
        :param image_path: RGB image [height, width, 3]
        :return: {list: N}{tuple: 1}('np.uint8' (C, 1, 2)), N - masks, C - coords
        """
        masks = self.infer_masks(image_path)
        polys = []
        for i in range(masks.shape[-1]):
            mask = masks[:, :, i]
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            polys.append(contours)
        return polys
