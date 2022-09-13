import utils
from mst_solver import MST_Solver
from puzzle import *
from puzzle_piece_detector.inference_callable import Inference
from utils import print_sol, masks_in_scale, image_in_scale


def main():
    is_inference_type = True
    segmentation_method = (is_inference_type, Inference.infer_masks_and_blur)
    # is_inference_type = False
    # segmentation_method = (is_inference_type, utils.infer_using_saturation_and_hue)
    weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale.jpg'
    if segmentation_method[0]:
        inference = Inference(weights_path)
        masks = segmentation_method[1](inference, image_path)

    else:
        masks = segmentation_method[1](image_path)
    image = skimage.io.imread(image_path)
    pieces = pieces_from_masks(masks, image)
    iou, n_facets, n_pieces, n_side_pieces, n_middle_pieces = evaluate_edge_compatibility(pieces, Facet.iou)
    piece_def = (n_pieces, n_side_pieces, n_middle_pieces)

    scale = 0.1
    sml_image = image_in_scale(image, scale)
    sml_masks = masks_in_scale(masks, scale)
    sml_pieces = pieces_from_masks(sml_masks, sml_image)
    mgc, n_facets, n_pieces, n_side_pieces, n_middle_pieces = evaluate_edge_compatibility(sml_pieces, Facet.mgc)
    cmp = calc_cmp_from_iou_and_mgc(iou, mgc, n_facets, n_pieces, pieces)

    edges_by_iou = sort_and_filter(n_pieces, n_facets, 0, iou, descending=True)
    edges_by_mgc = sort_and_filter(n_pieces, n_facets, 0, mgc, descending=False)
    edges_by_cmp = sort_and_filter(n_pieces, n_facets, 0, cmp, descending=False)

    # print_pieces(pieces, "pieces")
    # print_pieces(sml_pieces, "sml_pieces")
    # print_facets(pieces, "facets")
    # print_facets(sml_pieces, "sml_facets")
    #
    # print_figures_with_weights_to_folder(edges_by_iou, iou, pieces, 'iou')
    # print_figures_with_weights_to_folder(edges_by_mgc, mgc, sml_pieces, 'mgc')
    # print_figures_with_weights_to_folder(edges_by_cmp, cmp, pieces, 'cmp')

    solution_iou = MST_Solver(piece_def, edges_by_iou, iou).solveMST()
    solution_mgc = MST_Solver(piece_def, edges_by_mgc, mgc).solveMST()
    solution_cmp = MST_Solver(piece_def, edges_by_cmp, cmp).solveMST()
    print_sol(solution_iou, pieces, "iou")
    print_sol(solution_mgc, pieces, "mgc")
    print_sol(solution_cmp, pieces, "cmp")


if __name__ == '__main__':
    main()
