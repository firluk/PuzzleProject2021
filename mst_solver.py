# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph

# Class to represent a graph
import numpy as np

import block as bd


class MST_Solver:

    def __init__(self, piece_def, edges, score_mat):
        # TODO: asc or desc order of the final scoring system?
        # TODO: currently adapted for non-decreasing order
        self.V = piece_def[0]  # No. of puzzle pieces
        self.n_side = piece_def[1]  # TODO: use for frame calculation
        self.n_middle = piece_def[2]
        self.n_corner = 4
        self.graph = edges  # sorted edges by score
        self.score_mat = score_mat

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)

    def union(self, parent, rank, x, y, i, blocks):
        # def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        # print(rank[xroot], xroot, rank[yroot], yroot)

        x_block = self.findBlock(blocks, xroot)
        y_block = self.findBlock(blocks, yroot)
        if rank[xroot] < rank[yroot]:
            # TODO: fix block indecies!!!
            block = bd.joinBlocks(blocks[x_block], blocks[y_block],
                                  self.graph[i][0], self.graph[i][2], self.graph[i][1], self.graph[i][3],
                                  self.score_mat)
            if block is False:
                return block
            else:
                blocks[x_block].block = block[0]
                del blocks[y_block]
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            block = bd.joinBlocks(blocks[x_block], blocks[y_block],
                                  self.graph[i][0], self.graph[i][2], self.graph[i][1], self.graph[i][3],
                                  self.score_mat)
            if block is False:
                return block
            else:
                blocks[y_block].block = block[0]
                blocks[y_block].id = xroot
                del blocks[x_block]
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            block = bd.joinBlocks(blocks[x_block], blocks[y_block],
                                  self.graph[i][0], self.graph[i][2], self.graph[i][1], self.graph[i][3],
                                  self.score_mat)
            if block is False:
                return block
            else:
                blocks[x_block].block = block[0]
                # blocks[x_block].id = xroot
                del blocks[y_block]
            parent[yroot] = xroot
            rank[xroot] += 1

    def validateEdge(self, i, validationMat):

        while (validationMat[self.graph[i][0]][self.graph[i][2]] != 0) and \
                (validationMat[self.graph[i][1]][self.graph[i][3]] != 0):
            i = i + 1
        return i, self.graph[i]

    def findBlock(self, blocks, ind):

        for i, block in enumerate(blocks):
            if block.id == ind:
                return i

    def print_solution(self, assembled_block):
        for row in assembled_block:
            for cell in row:
                print("piece %d facet %d" % (cell.piece_ind, cell.facet_piece_ind))

    # The main function to construct MST using Kruskal's algorithm
    def solveMST(self):

        # result = []  # This will store the resultant MST

        # An index variable, used for sorted edges
        i = 0

        # An index variable, used for result[]
        e = 0

        parent = []
        rank = []
        validation_mat = np.zeros((self.V, 4), dtype=int)
        blocks = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
            blocks.append(bd.Block(node))  # create blocks of single piece each TODO: complete the code

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            i, [p1, p2, fp1, fp2] = self.validateEdge(i, validation_mat)  # returns edge with 2 available facets

            x = self.find(parent, p1)
            y = self.find(parent, p2)

            # If including this edge doesn't
            # cause cycle, include it in result
            # and increment the indexof result
            # for next edge
            if x != y:
                print(parent, rank)
                # return list of edges involved indirectly in the block or empty list
                # e = e + 1
                # self.union(parent, rank, x, y)
                # result.append([p1, p2, fp1, fp2])
                # validation_mat[p1][fp1] = 1
                # validation_mat[p2][fp2] = 1
                if self.union(parent, rank, x, y, i, blocks) is not False:
                    e = e + 1
                    # result.append([p1, p2, fp1, fp2])
                    validation_mat[p1][fp1] = 1
                    validation_mat[p2][fp2] = 1

            i = i + 1
        self.print_solution(blocks[0].block)
        # Else discard the edge

        # minimumCost = 0
        # print("Edges in the constructed MST")
        # for p1, p2, _, _ in result:
        #     #     minimumCost += weight
        #     print("%d -- %d" % (p1, p2))

        # print("Minimum Spanning Tree", minimumCost)


# Driver code

# edges = [[0, 1, 2, 2], [0, 2, 2, 0], [2, 3, 1, 3], [1, 2, 3, 0],
#          [1, 3, 2, 0], [2, 3, 0, 0], [0, 3, 1, 0], [0, 1, 1, 3],
#          [4, 5, 1, 1], [3, 4, 2, 2]]
# # [4, 5, 1, 1], [3, 4, 2, 2]
# cmp = np.load("cmp.npy")
# g = MST_Solver(6, edges, cmp)
#
# # n_facets = 4
# # n_pieces = 24
# # n_side_pieces = 12
# # n_middle_pieces = 8
#
# # Function call
# g.solveMST()

# This code is contributed by Neelam Yadav

###### mocks and driver
def mock1():
    #
    # Minimalistic mock having only 5 edges, all of which are part of the solution
    #
    # Facets of piece 'p'
    #
    #    0
    # 1 [p] 3
    #    2
    #
    # Suppose the Puzzle is 2 x 3
    #
    # [0] [1]
    # [2] [3]
    # [4] [5]
    #
    # Test 1:   Edges:
    #
    # [0]9[1] ((0,1,3,1),9),
    #  8   1  ((0,2,2,0),8), ((1,3,2,0),1),
    # [2]4[3] ((2,3,3,1),4),
    #  1   7  ((2,4,2,0),7), ((3,5,2,0),1),
    # [4]6[5] ((4,5,3,1),6),
    #
    edges_and_vals = \
        [
            ((0, 1, 3, 1), 9),
            ((0, 2, 2, 0), 8), ((1, 3, 2, 0), 1),
            ((2, 3, 3, 1), 4),
            ((2, 4, 2, 0), 7), ((3, 5, 2, 0), 1),
            ((4, 5, 3, 1), 6)
        ]
    n_pieces = 6
    n_facets = 4
    n_side = 2
    n_middle = 0

    score_mat = np.zeros((n_pieces, n_pieces, n_facets, n_facets))
    for i, (edge, val) in enumerate(edges_and_vals):
        score_mat[edge] = val
    edges = [edge for (edge, val) in (sorted(edges_and_vals, key=lambda edge_val: edge_val[1], reverse=True))]

    return (n_pieces, n_side, n_middle), edges, score_mat


def main():
    piece_def, edges, score_mat = mock1()
    solver = MST_Solver(piece_def, edges, score_mat)
    solver.solveMST()


if __name__ == '__main__':
    main()
