import matplotlib.pyplot as plt

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
    # weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale.jpg'
    weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale_3_pieces_corner_side_mid.jpg'
    # weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale_4_tab_3_tab_1_blank.jpg'
    # weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale_4_tab.jpg'  # single piece
    inference = Inference(weights_path)
    masks = inference.infer_masks_and_watershed(image_path)
    # masks = np.load("masks.npy")
    # masks = inference.infer_masks(image_path)
    # masks = inference.infer_masks(image_path)
    # via_region_data_json_path, filename = 'dataset/12pieces/val/via_region_data.json', 'front_white.jpg'
    # masks = utils.masks_from_via_region_data(via_region_data_json_path, filename)
    # image_path = os.path.join(os.path.split(via_region_data_json_path)[0], filename)

    image = cv.imread(image_path)
    scale = 1
    masks = masks_in_scale(masks, scale)
    image = image_in_scale(image, scale)

    pieces = pieces_from_masks(masks, image)

    piece = pieces[1]
    piece.facets

    # n_flats
    # n_inners
    # n_total

    print_pieces(pieces)
    print_facets(pieces)

    n_pieces = len(pieces)
    n_facets = 4

    iou = np.zeros((n_pieces, n_pieces, n_facets, n_facets))
    mgc = np.zeros((n_pieces, n_pieces, n_facets, n_facets))
    cmp = np.zeros((n_pieces, n_pieces, n_facets, n_facets))
    length_for_comparison = 25

    for p1idx, p1 in enumerate(pieces):
        for p2idx, p2 in enumerate(pieces):
            if p1idx < p2idx:
                for f1idx, f1 in enumerate(p1.facets):
                    if f1.type is Facet.Type.FLAT:
                        continue
                    for f2idx, f2 in enumerate(p2.facets):
                        if f2.type is Facet.Type.FLAT:
                            continue
                        iou[p1idx, p2idx, f1idx, f2idx] = Facet.iou(f1, f2)
                        # mgc[p1idx, p2idx, f1idx, f2idx] = Facet.mgc(f1, f2, length_for_comparison)
                        # cmp[p1idx, p2idx, f1idx, f2idx] = Facet.compatibility(f1, f2, length_for_comparison)

    # sort and filter in descending order
    edges_by_mgc = sort_and_filter(n_pieces, n_facets, 0, mgc, descending=True)
    edges_by_iou = sort_and_filter(n_pieces, n_facets, 0, iou, descending=False)
    edges_by_cmp = sort_and_filter(n_pieces, n_facets, 0, cmp, descending=True)

    print_figures_with_weights(edges_by_mgc, mgc, pieces, 'mgc')
    print_figures_with_weights(edges_by_iou, iou, pieces, 'iou')


def print_pieces(pieces):
    fig = plt.figure()

    for i in range(len(pieces)):
        factor = int(np.ceil(np.sqrt(len(pieces))))
        ax = fig.add_subplot(factor, factor, i + 1)
        plt.imshow(pieces[i].cropped_image)
        ax.set_title(f'{i}')
        ax.axis('off')
    plt.savefig(f'plots/pieces.png')
    plt.close(fig)


def print_facets(pieces):
    fig = plt.figure()
    for i in range(len(pieces)):
        factor = int(np.ceil(np.sqrt(len(pieces))))
        ax = fig.add_subplot(factor, factor, i + 1)
        img = np.zeros_like(pieces[i].cropped_image)
        facet_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]).astype(np.uint8)
        for fi in range(len(pieces[i].facets)):
            mask = pieces[i].facets[fi].facet_mask
            img[mask, :] = facet_colors[fi, :]
        plt.imshow(img)
        ax.set_title(f'{i}')
        ax.axis('off')

    plt.savefig(f'plots/facets.png')
    plt.close(fig)


def print_figures_with_weights(edges, arr, pieces, output_dir):
    for edge in edges:
        p1 = edge[0]
        p2 = edge[1]
        f1 = edge[2]
        f2 = edge[3]

        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(pieces[p1].cropped_image)
        ax.set_title(f'{p1}')

        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(pieces[p2].cropped_image)
        ax.set_title(f'{p2}')

        ax = fig.add_subplot(2, 2, 3)
        plt.imshow(pieces[p1].facets[f1].facet_mask)
        ax.set_title(f'{f1}')

        ax = fig.add_subplot(2, 2, 4)
        plt.imshow(pieces[p2].facets[f2].facet_mask)
        ax.set_title(f'{f2}')

        plt.savefig(f'plots/{output_dir}/{format(arr[p1, p2, f1, f2], ".3f")}_{p1}_{p2}_{f1}_{f2}.png')
        plt.close()


def sort_and_filter(n_pieces, n_facets, filter_val, weights, descending=True):
    sort_idx = np.argsort(weights, axis=None)
    if descending:
        sort_idx = np.flip(sort_idx)  # descending order using flat
    mask = weights.flat > filter_val  # filter mask,
    filtered_idx = sort_idx[mask[sort_idx]]  # cutting according to filter
    # get indices
    edges = np.transpose(np.vstack(np.unravel_index(filtered_idx, (n_pieces, n_pieces, n_facets, n_facets))))
    return edges


if __name__ == '__main__':
    main()
