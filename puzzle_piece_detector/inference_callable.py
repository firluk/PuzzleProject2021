"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


Derived from Balloon example by Waleed Abdulla as stated above.

------------------------------------------------------------

Usage:


    # Apply color splash to an image
    inference = Inference()
    inference.infer_masks() // or any of the other methods of the class

"""

import os
import sys
from warnings import filterwarnings

import cv2
import cv2 as cv
import numpy as np
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

    def infer_masks(self, image_path, as_bool=False):
        """
        Detect objects in image, using given model, returns binary bitmap masks
        :param as_bool: flag return values be boolean
        :param image_path: path to RGB image [height, width, 3]
        :return: np boolean array [height, width, N], N - number of detected instances
        """
        # Run model detection and generate the color splash effect
        print(f"Running on {image_path}")
        image = skimage.io.imread(image_path)
        # Detect objects
        r = self.model.detect([image], verbose=1)[0]
        masks = r['masks'].astype(np.uint8) * 255

        return masks.astype(np.bool_) if as_bool else masks

    def infer_masks_and_blur(self, image_path, as_bool=False):
        """
        Detect objects in image, using given model and blur using median blur, returns binary bitmap masks
        :param as_bool: flag return values be boolean
        :param image_path: path to RGB image [height, width, 3]
        :return: np boolean array [height, width, N], N - number of detected instances
        """
        # Run model detection and generate the color splash effect
        print(f"Running on {image_path}")
        image = skimage.io.imread(image_path)
        # Detect objects
        r = self.model.detect([image], verbose=1)[0]
        masks = r['masks'].astype(np.uint8) * 255

        _, _, stats, _ = cv.connectedComponentsWithStats(masks[:, :, 0])

        ksize_in_relation_to_width = int(stats[1, cv.CC_STAT_WIDTH] * 0.05) | 1

        for i in range(masks.shape[-1]):
            mask = masks[:, :, i]
            masks[:, :, i] = cv2.medianBlur(mask, ksize_in_relation_to_width)

        return masks.astype(np.bool_) if as_bool else masks

    def infer_masks_watershed_and_blur(self, image_path, as_bool=False):
        masks = self.infer_masks_and_blur(image_path)
        inferred_thresh = np.sum(masks, axis=2, keepdims=True).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        inferred_thresh = cv.morphologyEx(inferred_thresh, cv.MORPH_ERODE, kernel, iterations=2)

        img = skimage.io.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 254, 255, cv.THRESH_BINARY_INV)
        # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # noise removal
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        # Using inferred sure foreground area
        sure_fg = inferred_thresh
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        # Marker labelling
        _, markers, stats, _ = cv.connectedComponentsWithStats(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]
        vals = np.unique(markers)
        masks = np.zeros_like(masks, dtype=np.uint8)

        # ksize_in_relation_to_width = int(stats[1, cv.CC_STAT_WIDTH] * 0.10) | 1
        ksize_in_relation_to_width = int(stats[1, cv.CC_STAT_WIDTH] * 0.05) | 1

        for i, val in enumerate(vals[vals > 1]):
            mask = masks[:, :, i]
            mask[markers == val] = 255
            masks[:, :, i] = cv2.medianBlur(mask, ksize_in_relation_to_width)
            # plt.imshow(masks[:, :, i])
            # plt.show()

        return masks.astype(np.bool_) if as_bool else masks

    def infer_polys(self, image_path):
        """
        Detect objects in image, using given model, returns a list of polygons in form of coordinates (x,y).
        :param image_path: RGB image [height, width, 3]
        :return: {list: N}{tuple: 1}('np.uint8' (C, 1, 2)), N - masks, C - coords
        """
        masks = self.infer_masks_and_blur(image_path)
        polys = []
        for i in range(masks.shape[-1]):
            mask = masks[:, :, i]
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            polys.append(contours)
        return polys
