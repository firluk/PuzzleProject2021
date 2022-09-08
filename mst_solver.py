# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph

from collections import defaultdict

# Class to represent a graph
import numpy as np
import block as bd


class Graph:

    def __init__(self, numOfPieces, edges):
        # TODO: asc or desc order of the final scoring system?
        # TODO: currently adapted for non-decreasing order
        self.V = numOfPieces  # No. of puzzle pieces
        self.graph = edges  # sorted edges by score

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
        print(rank[xroot], xroot, rank[yroot], yroot)

        x_block = self.findBlock(blocks, xroot)
        y_block = self.findBlock(blocks, yroot)
        if rank[xroot] < rank[yroot]:
            # TODO: fix block indecies!!!
            block = bd.joinBlocks(blocks[x_block], blocks[y_block],
                                  self.graph[i][0], self.graph[i][2], self.graph[i][1], self.graph[i][3])
            if block is False:
                return block
            else:
                blocks[x_block] = block
                del blocks[y_block]
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            block = bd.joinBlocks(blocks[x_block], blocks[y_block],
                                  self.graph[i][0], self.graph[i][2], self.graph[i][1], self.graph[i][3])
            if block is False:
                return block
            else:
                blocks[y_block] = block
                del blocks[x_block]
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            block = bd.joinBlocks(blocks[x_block], blocks[y_block],
                                  self.graph[i][0], self.graph[i][2], self.graph[i][1], self.graph[i][3])
            if block is False:
                return block
            else:
                blocks[x_block] = block
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
            if block.block_id == ind:
                return i

                # The main function to construct MST using Kruskal's
    # algorithm

    def KruskalMST(self):

        result = []  # This will store the resultant MST

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
            blocks.append(bd.Block(node))   # create blocks of single piece each TODO: complete the code

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            i, [p1, p2, fp1, fp2] = self.validateEdge(i, validation_mat)     # returns edge with 2 available facets

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
                    result.append([p1, p2, fp1, fp2])
                    validation_mat[p1][fp1] = 1
                    validation_mat[p2][fp2] = 1

            i = i + 1
        # Else discard the edge

        # minimumCost = 0
        # print("Edges in the constructed MST")
        for p1, p2, _, _ in result:
            #     minimumCost += weight
            print("%d -- %d" % (p1, p2))

        # print("Minimum Spanning Tree", minimumCost)


# Driver code

edges = [[0, 1, 2, 2], [0, 2, 2, 0], [2, 3, 1, 3], [1, 2, 3, 0],
         [1, 3, 2, 0], [2, 3, 0, 0], [0, 3, 1, 0], [0, 1, 1, 3],
         [4, 5, 1, 1], [3, 4, 2, 2]]
# [4, 5, 1, 1], [3, 4, 2, 2]
g = Graph(6, edges)

# Function call
g.KruskalMST()

# This code is contributed by Neelam Yadav
