import utils
from mst_solver import MST_Solver
from puzzle import *
from utils import print_sol


def main():
    # is_inference_type = True
    # segmentation_method = (is_inference_type, Inference.infer_masks_and_blur)
    is_inference_type = False
    segmentation_method = (is_inference_type, utils.infer_using_saturation_and_hue)
    weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale.png'
    pieces, masks = segment_to_masks_and_extract_pieces(weights_path, image_path, segmentation_method)


    cmp, iou, mgc, n_facets, n_pieces, n_side_pieces, n_middle_pieces = evaluate_edge_compatibility(pieces)
    piece_def = (n_pieces, n_side_pieces, n_middle_pieces)
    # edges_by_mgc = sort_and_filter(n_pieces, n_facets, 0, mgc, descending=True)
    edges_by_iou = sort_and_filter(n_pieces, n_facets, 0, iou, descending=False)
    # edges_by_cmp = sort_and_filter(n_pieces, n_facets, 0, cmp, descending=True)

    # print_figures_with_weights_to_folder(edges_by_mgc, mgc, pieces, 'mgc')
    print_figures_with_weights_to_folder(edges_by_iou, iou, pieces, 'iou')

    solution = MST_Solver(piece_def, edges_by_iou, iou).solveMST()
    # solution = MST_Solver(piece_def, edges_by_mgc, mgc).solveMST()
    # solution = MST_Solver(piece_def, edges_by_cmp, cmp).solveMST()
    print_sol(solution, pieces)


if __name__ == '__main__':
    main()
