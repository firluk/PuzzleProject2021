from mst_solver import MST_Solver
from puzzle import *


def main():
    segmentation_method = Inference.infer_masks_and_blur
    # weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale.png'
    weights_path, image_path = './weights/mask_rcnn_puzzle.h5', 'https://image.shutterstock.com/mosaic_250/186843688/1721697922/stock-vector-puzzle-jigsaw-puzzle-icon-vector-design-template-1721697922.jpg'
    pieces, masks = segment_to_masks_and_extract_pieces(weights_path, image_path, segmentation_method)
    cmp, iou, mgc, n_facets, n_pieces, n_side_pieces, n_middle_pieces = evaluate_edge_compatibility(pieces)
    piece_def = (n_pieces, n_side_pieces, n_middle_pieces)
    # sort and filter in descending order
    edges_by_mgc = sort_and_filter(n_pieces, n_facets, 0, mgc, descending=True)
    edges_by_iou = sort_and_filter(n_pieces, n_facets, 0, iou, descending=False)
    edges_by_cmp = sort_and_filter(n_pieces, n_facets, 0, cmp, descending=True)
    # print_figures_with_weights_to_folder(edges_by_mgc, mgc, pieces, 'mgc')
    print_figures_with_weights_to_folder(edges_by_iou, iou, pieces, 'iou')
    # TODO: user selects edges type
    solution = MST_Solver(piece_def, edges_by_iou, iou).solveMST()


if __name__ == '__main__':
    main()
