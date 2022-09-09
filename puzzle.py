import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from facet import Facet
from main import main
from piece import Piece, pieces_from_masks, masks_in_scale, image_in_scale
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


def evaluate_edge_compatibility(pieces):
    n_side_pieces = n_middle_pieces = n_corner_pieces = 0
    for piece in pieces:
        if piece.type is Piece.Type.SIDE:
            n_side_pieces = n_side_pieces + 1
        elif piece.type is Piece.Type.MIDDLE:
            n_middle_pieces = n_middle_pieces + 1
        elif piece.type is Piece.Type.CORNER:
            n_corner_pieces = n_corner_pieces + 1
        else:
            raise
    n_pieces = len(pieces)
    n_facets = 4
    # print_pieces(pieces)
    # print_facets(pieces)
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
                        mgc[p1idx, p2idx, f1idx, f2idx] = Facet.mgc(f1, f2, length_for_comparison)
                        cmp[p1idx, p2idx, f1idx, f2idx] = Facet.compatibility(f1, f2, length_for_comparison)
                        # cmp[p1idx, p2idx, f1idx, f2idx] = compatibility_func()(mgc[p1idx, p2idx, f1idx, f2idx],
                        #                                                        iou[p1idx, p2idx, f1idx, f2idx])
    return cmp, iou, mgc, n_facets, n_pieces, n_side_pieces, n_middle_pieces


def segment_to_masks_and_extract_pieces(weights_path, image_path, segmenting_method):
    inference = Inference(weights_path)
    masks = segmenting_method(inference, image_path)
    image = cv.imread(image_path)
    scale = 1
    masks = masks_in_scale(masks, scale)
    image = image_in_scale(image, scale)
    pieces = pieces_from_masks(masks, image)
    return pieces, masks


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


def print_figures_with_weights_to_folder(edges, arr, pieces, output_dir):
    for edge in edges:
        p1, p2, f1, f2 = edge[0], edge[1], edge[2], edge[3]

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


def paint_facets_by_type(masks, pieces):
    width, height, _ = masks.shape
    masks_with_facets = np.ones((width, height, 3), dtype=np.uint8) * 255
    for piece in pieces:
        img = np.ones_like(piece.cropped_image) * 255
        facet_colors = dict()
        facet_colors[Facet.Type.FLAT] = np.array([0, 0, 0], dtype=np.uint8)
        facet_colors[Facet.Type.TAB] = np.array([0, 255, 0], dtype=np.uint8)
        facet_colors[Facet.Type.BLANK] = np.array([255, 0, 0], dtype=np.uint8)
        for fi in range(len(piece.facets)):
            facet = piece.facets[fi]
            mask = facet.facet_mask
            img[mask, :] = facet_colors[facet.type]
        height, width, _ = piece.cropped_image.shape
        left, top = piece.left, piece.top
        left_width, top_height = left + width, top + height
        masks_with_facets[top:top_height, left:left_width] = img
    return masks_with_facets


def paint_facets_distinct(masks, pieces):
    width, height, _ = masks.shape
    masks_with_facets = np.ones((width, height, 3), dtype=np.uint8) * 255
    for piece in pieces:
        img = np.ones_like(piece.cropped_image) * 255
        facet_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]]).astype(np.uint8)
        for fi in range(len(piece.facets)):
            mask = piece.facets[fi].facet_mask
            img[mask, :] = facet_colors[fi, :]
        height, width, _ = piece.cropped_image.shape
        left, top = piece.left, piece.top
        left_width, top_height = left + width, top + height
        masks_with_facets[top:top_height, left:left_width] = img
    return masks_with_facets
