from mst_solver import MST_Solver
from puzzle import *


def print_sol(solution, pieces):
    # TODO: move to puzzle.py
    for i, sol in enumerate(solution):
        fig = plt.figure()

        for j, cell in enumerate(sol.block.flatten()):
            ax = fig.add_subplot(sol.block.shape[0], sol.block.shape[1], j + 1)
            if cell is not None:
                if cell.facet_piece_ind == 1:
                    img = np.rot90(pieces[cell.piece_ind].cropped_image, -1)
                elif cell.facet_piece_ind == 2:
                    img = np.rot90(pieces[cell.piece_ind].cropped_image, 2)
                elif cell.facet_piece_ind == 3:
                    img = np.rot90(pieces[cell.piece_ind].cropped_image, 1)
                else:
                    img = pieces[cell.piece_ind].cropped_image
                plt.imshow(img)
                ax.axis('off')
            else:
                plt.imshow(np.array([0]))
                ax.axis('off')
        plt.savefig(f'plots/block{i}.png')
        plt.close(fig)

def main():
    segmentation_method = Inference.infer_masks_and_blur
    # weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/full_downscale.png'
    weights_path, image_path = './weights/mask_rcnn_puzzle.h5', './plots/small_img.png'
    # weights_path, image_path = './weights/mask_rcnn_puzzle.h5', 'https://image.shutterstock.com/mosaic_250/186843688/1721697922/stock-vector-puzzle-jigsaw-puzzle-icon-vector-design-template-1721697922.jpg'
    pieces, masks = segment_to_masks_and_extract_pieces(weights_path, image_path, segmentation_method)
    cmp, iou, mgc, n_facets, n_pieces, n_side_pieces, n_middle_pieces = evaluate_edge_compatibility(pieces)
    piece_def = (n_pieces, n_side_pieces, n_middle_pieces)
    # sort and filter in descending order
    edges_by_mgc = sort_and_filter(n_pieces, n_facets, 0, mgc, descending=True)
    edges_by_iou = sort_and_filter(n_pieces, n_facets, 0, iou, descending=False)
    edges_by_cmp = sort_and_filter(n_pieces, n_facets, 0, cmp, descending=True)
    # print_figures_with_weights_to_folder(edges_by_mgc, mgc, pieces, 'mgc')
    # print_figures_with_weights_to_folder(edges_by_iou, iou, pieces, 'iou')
    # TODO: user selects edges type
    solution = MST_Solver(piece_def, edges_by_mgc, mgc).solveMST()
    print_sol(solution, pieces)
    print("stop")


if __name__ == '__main__':
    main()
