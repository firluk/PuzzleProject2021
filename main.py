from mst_solver import MST_Solver
from puzzle import *


def main():
    pieces = segment_to_masks_and_extract_pieces('./weights/mask_rcnn_puzzle.h5', './plots/full_downscale.png')
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
