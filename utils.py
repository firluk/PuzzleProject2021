import json
import os

import numpy as np
import skimage.draw
import skimage.io


def masks_from_via_region_data(via_region_data_json_path, filename):
    """
    Given a via_region_data.json file path and file name, returns bit mask of drawn polygon according to coordinates.
    :param via_region_data_json_path: path to VIA annotations file, containing coordinates of bit mask.
    :param filename: path to RGB image [height, width, 3]
    :return: np boolean array [height, width, N], N - number of detected instances
    """
    annotations = json.load(open(via_region_data_json_path))

    annotations = list(annotations.values())  # don't need the dict keys
    annotations = [a for a in annotations if a['regions']]
    annotation = [a for a in annotations if a['filename'] == filename][0]

    dataset_dir, _ = os.path.split(via_region_data_json_path)
    image_path = os.path.join(dataset_dir, filename)
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]
    masks = np.zeros((height, width, len(annotation['regions'])), dtype=np.uint8)
    for i, r in enumerate(annotation['regions']):
        shape_attributes = r['shape_attributes']
        all_points_x = shape_attributes['all_points_x']
        all_points_y = shape_attributes['all_points_y']
        rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
        masks[rr, cc, i] = 1

    return masks
