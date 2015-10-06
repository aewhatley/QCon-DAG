"""
TreeBuilder

Class to build an unrooted tree
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ["TreeBuilder", "InCompatableError"]

# *************************************************************************

from   numpy    import random
from   operator import itemgetter
import sys

from   TreeUtilities.Bipartitions import Bipartitions
from   TreeUtilities.Tree         import UnrootedTree, Edge, Inner, Leaf, TreeVisitor



# *************************************************************************

class InCompatableError(Exception):
    """
    Exception class for the case when an unambiguous location to split
    can't be found
    """

    def __init__(self, commonAncestors):
        self.commonAncestors = commonAncestors

    def __str__(self):
        return ','.join(str(l) for l in self.commonAncestors)

# *************************************************************************

class TreeBuilder(object):
    """
    Build an unrooted tree (not necessairly binary).
    Input can either be a a set of bipartitions or a starting tree.

    The bipartitions (if given) must be known to be compatable
    but not necessairly complete.
    """

    def __init__(self, input):
        """
        Input to the builder can be either a tree or a set of bipartitions
        """

        self.diameter = 0

        if type(input) is Bipartitions:
            self.__tree = UnrootedTree()
            self.__InitBipartitions(input)
        elif type(input) is UnrootedTree:
            self.__tree = input
        else:
            raise TypeError(type(input))

    @property
    def tree(self):
        """Return the tree built"""
        return self.__tree;

    def AddLeafToTree(self, id, diameter):
        """
        Add a single leaf to the tree. The position is guided
        by the diameter parameter which indicates the precentage
        of the maximum possible tree diameter to use.
        """

        # Handle the cases where the tree is empty (or a single node)
        if len(self.__tree.vertices) == 0:
            Leaf(label=id, tree=self.__tree)
        elif len(self.__tree.vertices) == 1:
            Edge(nodes=[self.__tree.vertices[0],
                        Leaf(label=id, tree=self.__tree)],
                 tree=self.__tree)
        else:
            # Find the edges which will (and will not) increase the diameter
            (will, willnot)  = self.__PartitionEdges()

            # if no unmarked edges exist or we want to increase diameter
            # randomly select an edge to split

            if not len(willnot) or random.random() > (1.0 - diameter):
                esplit = random.choice(will)
            else:
                esplit = random.choice(willnot)

            # Add new taxa splitting the edge
            self.__SplitEdge(esplit, id)

    def MakeBinary(self):
        """Convert vertices of degree >3 to degree 3 by breaking off pairs of edges"""

        # Loop through the non-binary vertices

        for nbv in (v for v in self.__tree.vertices if len(v.edges) > 3):

            # Split off pair of edges until the vertex is properly binary

            while len(nbv.edges) > 3:
                depths = sorted(((e.Depth(nbv), e) for e in nbv.edges), key=itemgetter(0))
                e1 = depths[0][1]
                e2 = depths[1][1]
                e1.RemoveNode(nbv)
                e2.RemoveNode(nbv)
                Edge(nodes=[nbv, Inner(edges=[e1, e2], tree=self.__tree)],
                     tree=self.__tree)

    def AddBranchLengths(self, mean):

        for e in self.tree.edges:
            e.length = 1 + 0.8 * random.exponential(mean)
#            l = random.exponential(mean)
#            e.length = random.exponential(mean)

    def __AddBipartition(self, bp):
        """
        Add a single bipartition to the tree

        bp contains a set of leaf objects not associated with any tree
        """

        # Find the sets of ancestors of the left and right sides of the
        # bipartition. New nodes are added as required

        lanc = self.__FindAncestors(bp.left)
        ranc = self.__FindAncestors(bp.right)

        # Find common ancestors

        canc = lanc.intersection(ranc)

        if len(canc):
            smaller = bp.left if len(bp.left) < len(bp.right) else bp.right
            self.__AddNewEdge(set(str(s) for s in smaller), canc)

    def __AddNewEdge(self, smallerLeafIds, commonAncestorNodes):
        """Add a new edge to one of the common ancestor nodes"""

        # For each common ancestor find the subtrees containing
        # members of the smaller set.

        edgesToMove = []
        for ancestorNode in commonAncestorNodes:
            for downEdge in ancestorNode.edges:
                childNode = downEdge.Other(ancestorNode)
                if childNode not in commonAncestorNodes:
                    childLeafIds = set(str(l) for l in childNode.Leaves(downEdge))
                    if len(childLeafIds.intersection(smallerLeafIds)):
                        edgesToMove.append((downEdge,ancestorNode))

        # Disconnect the subtrees from their common ancestors

        [edgeTuple[1].RemoveEdge(edgeTuple[0]) for edgeTuple in edgesToMove]

        # Find the new position for the edges looking for a node with the smallest number
        # of edges

        oldInner = min(commonAncestorNodes, key=lambda e : len(e.edges))
        newInner = Inner(edges = [e[0] for e in edgesToMove],
                         tree  = self.__tree)
        Edge(nodes=[newInner, oldInner],
             tree = self.__tree)

        # If any of the ancestor nodes now only have two edges, condense them out

        [self.__CondenseNode(n) for n in commonAncestorNodes if len(n.edges) == 2]

    def __CondenseNode(self, node):
        """Eliminate a node that is no longer needed"""

        upEdge = node.edges[0]
        downEdge = node.edges[1]
        upEdge.RemoveNode(node)
        downEdge.RemoveNode(node)
        self.__tree.Remove(node)

        upNode   = upEdge.nodes[0]
        downNode = downEdge.nodes[0]
        upNode.RemoveEdge(upEdge)
        downNode.RemoveEdge(downEdge)

        self.__tree.Remove(upEdge)
        self.__tree.Remove(downEdge)

        Edge(nodes=[upNode, downNode],
             tree=self.__tree)

    def __FindAncestors(self, leaves):
        """
        Finds all inner nodes in the path between the leaves.
        Leaves not found will be added.
        """

        leafIdSet = set(str(l) for l in leaves)

        # Find any new leafs

        newLeafIds  = leafIdSet.difference(self.__tree.leafSet)
        assert len(newLeafIds) != len(leafIdSet)

        oldLeafIds = leafIdSet - newLeafIds
        oldLeafObjs = [l for l in self.__tree.leaves if str(l) in oldLeafIds]

        # Only one id, add any nodes to it's parent

        if len(oldLeafObjs) == 1:
            parent = oldLeafObjs[0].edges[0].Other(oldLeafObjs[0])
            newLeafObjs = []
            for l in newLeafIds:
                leaf = Leaf(edge  = Edge(nodes=[parent], tree=self.__tree),
                            label = l,
                            tree  = self.__tree)
                newLeafObjs.append(leaf)

            inodes = self.__FindInnerNodes(oldLeafObjs + newLeafObjs)

        # Multiple old id's, find the overall path and add to the common node

        else:

            # Get the paths for the existing (old) leafs
            inodes = self.__FindInnerNodes(oldLeafObjs)

            assert not len(newLeafIds) or len(inodes) == 1

            # Add new nodes to the tree at the inode point
            [Leaf(edge  = Edge(nodes=[inodes[0]], tree=self.__tree),
                  label = l,
                  tree  = self.__tree) for l in newLeafIds]

        return inodes

    def __FindInnerNodes(self, leaves):
        """
        Find all the inner nodes that are in paths between the leaves passed
        """

        inodes = set()
        remainingLeaves = leaves

        while len(remainingLeaves) > 1:
            root = remainingLeaves.pop()
            visitor = _BuilderVisitor(remainingLeaves, inodes)
            self.__tree.DFS(visitor, edge=root.edges[0])

        return inodes

    def __InitBipartitions(self, bipartitions):

        # Rank the bipartitions in order of completeness, they will be used in reverse order
        # Not entirely sure this is required but is seems to make the process more effective
        rbl = sorted(bipartitions, key=lambda b: "{:05d}{}".format(100000-int(b), str(b)))

        if not len(rbl):                          # Empty list, no bipartitions
            return

        # Create an initial tree from the first bipartition
        self.__TreeFromBipartition(rbl.pop(0))

        # Loop through the remaining bipartitions
        try:
            [self.__AddBipartition(bp) for bp in rbl]
        except InCompatableError as err:
            sys.exit(err)

    def __PartitionEdges(self):
        """
        Determine which edges that, when split will increase the diameter
        of the tree. Return two lists, one of edges that will increase the
        diameter and one of edges that won't increase it.
        """

        # Initially assume edges won't increase diameter

        will = []                                 # List of edges that will increase diameter
        willnot = self.__tree.edges[:]            # List of edges that won't increase diameter

        # Starting with the first leaf in the list, find the verticies farthest from it

        leaf = next(v for v in self.__tree.vertices if isinstance(v,Leaf))
        farthest = _FarthestDFSVisitor()
        self.__tree.DFS(farthest, leaf.edges[0])

        # For each vertex found look for the farthest from them.

        paths = _PathsDFSVisitor()
        for leaf in farthest.leaves:
            paths.stack = [leaf.edges[0]]         # Prime the stack with the current edge
            self.__tree.DFS(paths, leaf.edges[0])

        if paths.maxstack > self.diameter:        # Save largest stack as diameter
            self.diameter = paths.maxstack

        # Move edges in the longest paths to the will list

        for p in paths.paths:
            for e in p:
                willnot.remove(e) if e in willnot else None
                will.append(e) if e not in will else None

        return (will, willnot)

    def __SplitEdge(self, edge, id):
        """
        Add a new taxa by splitting an existing edge
        """

        # Disconnect the edge

        nright = edge.nodes[0]
        nleft  = edge.nodes[1]
        nright.RemoveEdge(edge)
        nleft.RemoveEdge(edge)

        # Connect it all up

        Leaf(label=id, edge=edge, tree=self.__tree)
        Inner(edges=[Edge(nodes=[nleft], tree=self.__tree),
                     Edge(nodes=[nright], tree=self.__tree),
                     edge],
              tree=self.__tree)

    def __TreeFromBipartition(self, bp):
        """Generate an initial non-binary tree from a bipartition"""

        e     = Edge(tree=self.__tree)
        left  = Inner(edges=[e], tree=self.__tree)
        right = Inner(edges=[e], tree=self.__tree)

        [Edge(nodes=[Leaf(label=str(leaf),tree=self.__tree), left], tree=self.__tree)
         for leaf in bp.left]
        [Edge(nodes=[Leaf(label=str(leaf),tree=self.__tree), right], tree=self.__tree)
         for leaf in bp.right]

# *************************************************************************

class _BuilderVisitor(TreeVisitor):
    """
    Depth first search visitor class to save inner nodes along paths to the
    specified leafs
    """

    def __init__(self, leaves, inodes):
        self.__leaves = set(leaves)
        self.__inodes = inodes
        self.__stack = []

    def PreTreeVisit(self, t, incoming=None):
        """Called before exploring a subtree"""
        self.__stack.append(t)

    def PostTreeVisit(self, t, incoming=None):
        """Called after exploring a subtree"""
        self.__stack.pop()

    def VisitLeaf(self, l, incoming=None):
        """Called when visiting a leaf node"""
        if l in self.__leaves:
            [self.__inodes.add(n) for n in self.__stack]

# *************************************************************************

class _FarthestDFSVisitor(TreeVisitor):
    """
    Depth first search visitor class to determine the vertices farthest
    from the starting point
    """

    def __init__(self):
        self.maxstack = 0;
        self.stack = []
        self.leaves = []                          # Set of leaves farthest from root

    def PreTreeVisit(self, t, incoming=None):
        """Called before exploring a subtree"""
        self.stack.append(incoming)

    def PostTreeVisit(self, t, incoming=None):
        """Called after exploring a subtree"""
        self.stack.pop()

    def VisitLeaf(self, l, incoming=None):
        """Called when visiting a leaf node"""
        if len(self.stack) > self.maxstack:
            self.maxstack = len(self.stack)
            self.leaves = [l]
        elif len(self.stack) == self.maxstack:
            self.leaves.append(l)

# *************************************************************************

class _PathsDFSVisitor(TreeVisitor):
    """
    Depth first search visitor class to save the longest paths
    from the starting point
    """

    def __init__(self):
        self.maxstack = 0;
        self.stack = []
        self.paths = []                             # Set of longest paths

    def PreTreeVisit(self, t, incoming=None):
        """Called before exploring a subtree"""
        self.stack.append(incoming)

    def PostTreeVisit(self, t, incoming=None):
        """Called after exploring a subtree"""
        self.stack.pop()

    def VisitLeaf(self, l, incoming=None):
        """Called when visiting a leaf node"""
        if len(self.stack) > self.maxstack:
            self.maxstack = len(self.stack)
            self.paths = [self.stack[:]]
        elif len(self.stack) == self.maxstack:
            self.paths.append(self.stack[:])

# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
