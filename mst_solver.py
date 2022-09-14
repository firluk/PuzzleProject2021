import numpy as np
import block as bd


class MST_Solver:

    def __init__(self, piece_def, edges, score_mat):
        self.V = piece_def[0]  # No. of puzzle pieces
        # self.n_side = piece_def[1]  # for future work
        # self.n_middle = piece_def[2]    # for future work
        # self.n_corner = 4   # for future work
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
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)

        x_block = self.findBlock(blocks, xroot)
        y_block = self.findBlock(blocks, yroot)
        block = bd.joinBlocks(blocks[x_block], blocks[y_block],
                              self.graph[i][0], self.graph[i][2], self.graph[i][1], self.graph[i][3],
                              self.score_mat)
        if rank[xroot] < rank[yroot]:
            if block is False:
                return block
            else:
                blocks[y_block].block = block[0]
                del blocks[x_block]
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            if block is False:
                return block
            else:
                blocks[x_block].block = block[0]
                del blocks[y_block]
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            if block is False:
                return block
            else:
                blocks[x_block].block = block[0]
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

    # The main function to construct MST using Kruskal's algorithm
    def solveMST(self):
        # result = []  # This will store the resultant MST
        # An index variable, used for sorted edges
        i = 0
        # An index variable, used as stop-criterion for the While loop
        e = 0

        parent = []
        rank = []
        validation_mat = np.zeros((self.V, 4), dtype=int)
        blocks = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
            blocks.append(bd.Block(node))  # creates blocks of single piece each

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1 and i < self.graph.shape[0] - 1:
            # the index for next iteration
            i, [p1, p2, fp1, fp2] = self.validateEdge(i, validation_mat)  # returns edge with 2 available facets

            x = self.find(parent, p1)
            y = self.find(parent, p2)

            # If including this edge doesn't
            # cause cycle, include it in result
            # and increment the index of result
            # for next edge
            if x != y and self.union(parent, rank, x, y, i, blocks) is not False:
                e = e + 1

                validation_mat[p1][fp1] = 1
                validation_mat[p2][fp2] = 1
            i = i + 1
        return blocks
