# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph

from collections import defaultdict

# Class to represent a graph
import numpy as np


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
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def validateEdge(self, i, validationMat):

        while (validationMat[self.graph[i][0]][self.graph[i][2]] != 0) and \
                (validationMat[self.graph[i][1]][self.graph[i][3]] != 0):
            i = i + 1
        return self.graph[i]


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

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            p1, p2, fp1, fp2 = self.validateEdge(i, validation_mat)
            x = self.find(parent, p1)
            y = self.find(parent, p2)

            # If including this edge doesn't
            # cause cycle, include it in result
            # and increment the indexof result
            # for next edge
            if x != y:
                e = e + 1
                result.append([p1, p2, fp1, fp2])
                validation_mat[p1][fp1] = 1
                validation_mat[p2][fp2] = 1
                self.union(parent, rank, x, y)
        # Else discard the edge

        # minimumCost = 0
        # print("Edges in the constructed MST")
        for p1, p2, _, _ in result:
        #     minimumCost += weight
            print("%d -- %d" % (p1, p2))
        # print("Minimum Spanning Tree", minimumCost)

# Driver code

edges = [[0,1,2,2],[0,2,2,0],[2,3,1,3],[1,2,3,0],
         [1,3,2,0],[2,3,0,0],[0,3,1,0],[0,1,1,3]]
g = Graph(4,edges)

# Function call
g.KruskalMST()

# This code is contributed by Neelam Yadav
