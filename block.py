import numpy as np
import copy


class cellInBlock:

    def __init__(self, piece_ind = None, facet_piece_ind = None):

        self.piece_ind = piece_ind
        self.facet_piece_ind = facet_piece_ind

    def rotateCell(self, angle):
        # 1 - left; 2 - flip; 3 - right
        if self.piece_ind >= 0:
            if angle == 1:
                self.left()
            elif angle == 2:
                self.left()
                self.left()
            elif angle == 3:
                self.left()
                self.left()
                self.left()

    def left(self):
        # rotate cell left
        if self.facet_piece_ind == 0:
            self.facet_piece_ind = 3
        else:
            self.facet_piece_ind -= 1


class Block:

    def __init__(self, block_id):

        self.block = np.expand_dims(np.array([cellInBlock(block_id, 0)]), axis=1)
        self.id = block_id
        # self.multi_facet

    def rotateBlock(self, angle):
        # 1 - left; 2 - flip; 3 - right
        # TODO: instead if self.block return block - to avoid mutating the block property
        if angle == 1:
            block_tmp = copy.deepcopy(np.rot90(self.block, 1))
        # if angle == 1:
        #     self.block = np.rot90(self.block, 1)
        elif angle == 2:
            block_tmp = copy.deepcopy(np.rot90(self.block, 2))
            # self.block = np.rot90(self.block, 2)
        elif angle == 3:
            block_tmp = copy.deepcopy(np.rot90(self.block, -1))
            # self.block = np.rot90(self.block, -1)
        for cell in block_tmp.flatten():
            cell.rotateCell(angle)
        return block_tmp
        # for cell in self.block.flatten():
        #     cell.rotateCell(angle)

    def findCellInBlock(self, p):
        # ind = np.where(self.block.piece_ind == p)
        # return ind[0][0], ind[0][1], self.block.facet_piece_ind
        # TODO: (future) more efficient search algo

        for i, row in enumerate(self.block):
            for j, cell in enumerate(row):
                if cell.piece_ind == p:
                    return (i, j), cell.facet_pience_ind


def rotateBlockBeforeJoin(block, fp, fp_up, side):
    # side = 1 for block_a; side = 2 for block_b
    flip = 0
    if side == 2:
        flip = 2
    if fp == fp_up:
        return block.rotateBlock(3-flip)
    elif abs(fp - fp_up) == 1+flip:
        return block.rotateBlock(2)
    elif abs(fp - fp_up) == 2:
        return block.rotateBlock(1+flip)
    # return block


def validateMatch(score_mat, cell_a, cell_b):
    # score_mat is either IoU, Mahalanobis or CMP
    if cell_b.piece_ind >= cell_a.piece_ind:
        if score_mat[cell_a.piece_ind][cell_b.piece_ind][cell_a.facet_piece_ind][cell_b.facet_piece_ind] == 0:
            return False
        else:
            return True
    else:
        if score_mat[cell_b.piece_ind][cell_a.piece_ind][cell_b.facet_piece_ind][cell_a.facet_piece_ind] == 0:
            return False
        else:
            return True


def joinBlocks(block_a, block_b, p_a, fp_a, p_b, fp_b, score_mat):
    # in: indices of pieces and facets connected by valid edge
    # directions right/left/above/below are relative to block_a
    _, fp_a_up = block_a.findCellInBlock(p_a)    # TODO: has to be called again after rotation - redo!
    _, fp_b_up = block_b.findCellInBlock(p_b)    # TODO: has to be called again after rotation - redo!
    block_a_rotated = rotateBlockBeforeJoin(block_a, fp_a, fp_a_up, 1)
    block_b_rotated = rotateBlockBeforeJoin(block_b, fp_b, fp_b_up, 2)
    p_a_location, _ = block_a_rotated.findCellInBlock(p_a)    # TODO: has to be called again after rotation - redo!
    p_b_location, _ = block_b_rotated.findCellInBlock(p_b)    # TODO: has to be called again after rotation - redo!

    a_right_border = block_a_rotated.block.shape[1]
    a_lower_border = block_a_rotated.block.shape[0]
    # a_upper_border, a_left_border = 0, 0
    dist_left_p_a = p_a_location[1]
    dist_upper_p_a = p_a_location[0]
    dist_right_p_a = a_right_border - p_a_location[1]
    dist_lower_p_a = a_lower_border - p_a_location[0]

    b_right_border = block_b_rotated.block.shape[1]
    b_lower_border = block_b_rotated.block.shape[0]
    # b_upper_border, b_left_border = 0, 0
    dist_left_p_b = p_b_location[1]
    dist_upper_p_b = p_b_location[0]
    dist_right_p_b = b_right_border - p_b_location[1]
    dist_lower_p_b = b_lower_border - p_b_location[0]

    diff = dist_left_p_b - dist_left_p_a
    if diff > 1:
        block_a_rotated.block = np.hstack((np.empty((block_a_rotated.block.shape[0], abs(diff) - 1), cellInBlock), block_a_rotated.block))
    elif diff < 1:
        block_b_rotated.block = np.hstack((np.empty((block_b_rotated.block.shape[0], abs(diff) + 1), cellInBlock), block_b_rotated.block))

    diff = dist_right_p_b - dist_right_p_a
    if diff > -1:
        block_a_rotated.block = np.hstack((block_a_rotated.block, np.empty((block_a_rotated.block.shape[0], abs(diff) + 1), cellInBlock)))
    elif diff < -1:
        block_b_rotated.block = np.hstack((block_b_rotated.block, np.empty((block_b_rotated.block.shape[0], abs(diff) - 1), cellInBlock)))

    diff = dist_upper_p_b - dist_upper_p_a
    if diff > 0:
        block_a_rotated.block = np.vstack((np.empty((abs(diff), block_a_rotated.block.shape[1]), cellInBlock), block_a_rotated.block))
    elif diff < 0:
        block_b_rotated.block = np.vstack((np.empty((abs(diff), block_b_rotated.block.shape[1]), cellInBlock), block_b_rotated.block))

    diff = dist_lower_p_b - dist_lower_p_a
    if diff > 0:
        block_a_rotated.block = np.vstack((block_a_rotated.block, np.empty((abs(diff), block_a_rotated.block.shape[1]), cellInBlock)))
    elif diff < 0:
        block_b_rotated.block = np.vstack((block_b_rotated.block, np.empty((abs(diff), block_b_rotated.block.shape[1]), cellInBlock)))

    joined_block = np.empty(block_a_rotated.shape, cellInBlock)
    # TODO: overwrite block property of the target block object with joined_block
    used_edge_stack = []

    # for i, (cell_a, cell_b) in enumerate(zip(block_a_rotated.flatten(), block_b_rotated.flatten())):
    #     if cell_a is None and cell_b is not None:
    #         joined_block[i] = cell_b
    #     elif cell_a is not None and cell_b is None:
    #         joined_block[i] = cell_a
    #     elif cell_a is not None and cell_b is not None:
    #         return False

    for i, (row_a, row_b) in enumerate(zip(block_a_rotated, block_b_rotated)):
        for j, (cell_a, cell_b) in enumerate(zip(row_a, row_b)):
            if cell_a is not None and cell_b is not None:
                return False
            elif j < block_a_rotated.shape[1]:
                if cell_a is not None and block_b_rotated[i][j + 1] is not None:
                    if validateMatch(score_mat, cell_a, block_b_rotated[i][j + 1]) is False:
                        return False
                    else:
                        used_edge_stack.append(cell_a.piece_ind, block_b_rotated[i][j + 1].piece_ind,
                                               cell_a.facet_piece_ind, block_b_rotated[i][j + 1].facet_piece_ind)
                elif cell_b is not None and block_a_rotated[i][j + 1] is not None:
                    if validateMatch(score_mat, cell_b, block_a_rotated[i][j + 1]) is False:
                        return False
                    else:
                        used_edge_stack.append(cell_b.piece_ind, block_a_rotated[i][j + 1].piece_ind,
                                               cell_b.facet_piece_ind, block_a_rotated[i][j + 1].facet_piece_ind)
            elif i < block_a_rotated.shape[0]:
                if cell_a is not None and block_b_rotated[i + 1][j] is not None:
                    if validateMatch(score_mat, cell_a, block_b_rotated[i + 1][j]) is False:
                        return False
                    else:
                        used_edge_stack.append(cell_a.piece_ind, block_b_rotated[i + 1][j].piece_ind,
                                               cell_a.facet_piece_ind, block_b_rotated[i + 1][j].facet_piece_ind)
                elif cell_b is not None and block_a_rotated[i + 1][j] is not None:
                    if validateMatch(score_mat, cell_a, block_a_rotated[i + 1][j]) is False:
                        return False
                    else:
                        used_edge_stack.append(cell_b.piece_ind, block_a_rotated[i + 1][j].piece_ind,
                                               cell_b.facet_piece_ind, block_a_rotated[i + 1][j].facet_piece_ind)
            elif cell_a is None and cell_b is not None:
                joined_block[i][j] = cell_b
            elif cell_a is not None and cell_b is None:
                joined_block[i][j] = cell_a
    # TODO: maybe a global list of used edges so it can be updated with temporary used_edge_stack?

    return joined_block, used_edge_stack
