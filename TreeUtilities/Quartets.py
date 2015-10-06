"""
Class to expose the set of quartets for an unrooted tree

Quartets returned are ordered such that within each of the paris of taxa the
lowest collating value is first and the two pairs are ordered such that the
lowest collating pair is first.
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ["Quartets", "QuartetCountError"]

# *************************************************************************

from   operator           import attrgetter
from   pprint             import pprint as pp

from   TreeUtilities.Tree import *

# *************************************************************************

class QuartetCountError(Exception):
    """
    Mismatch in the number of quartets produced
    """
    
    # *****************************************************************
    
    def __init__(self, produced, truecount):
        self.produced = produced
        self.truecount = truecount

    def __str__(self):
        return("Produced {} instead of {}".format(self.produced, self.truecount))

# *************************************************************************

class Quartets(object):
    """
    Generate the list of quartets associated with a tree
    """

    # **************************************************

    def __init__(self, tree, startEdge=None):
        self.nQuartets = 0                        # Number of quartets generated so far
        self.tree = tree
        self.covers   = []                        # List of covered taxa
        self.startEdge = startEdge
         
    # **************************************************

    def __iter__(self):
        """
        Return the quartets one at a time
        """

        if not self.startEdge:
            if len(self.tree.edges) == 0:
                raise StopIteration
            edge = self.tree.edges[0]
        else:
            edge = self.startEdge

        for n in edge.nodes:
            yield from self._DFS(n, edge)

        # Clear visited - TODO might not need this... 
        [delattr(v, "visited") for v in self.tree.vertices]

    # **************************************************
            
    def Check(self):
        """
        Verify the number of quartets produced was correct
        """
        if self.nQuartets != self.tree.nQuartets:
            raise QuartetCountError(self.nQuartets, self.tree.nQuartets)

    # **************************************************
    
    def _DFS(self, root, incoming):
        """
        Non recursive depth first search
        """

        stack = []
        root.visited = False
        stack.append((root, incoming))
#        print("Push", root)

        while len(stack):
            
            node, inedge = stack[-1]
            
            if not node.visited and not type(node) is Leaf:
                node.visited = True
                for e in node.edges:
                    if not e is inedge:
                        child = e.Other(node)
                        child.visited = False
                        stack.append((child, e))
#                        print("Push", child)
            else:
                node, inedge = stack.pop()
#                print("Pop", node)

                if type(node) is Leaf:
                    continue

                # Get the sets of leaves incident to each of the three
                # edges associated with the node

                downedges = node.Other(inedge)
                print(len(downedges))
                left  = downedges[0].Other(node).Leaves(downedges[0])
                right = downedges[1].Other(node).Leaves(downedges[1])
                up = sorted([l for l in self.tree.leaves if not l in left and not l in right],
                            key=attrgetter('label'))

                # Loop through it all, do the up edge first so the cover list 
                # can be checked

#                if len(up) + len(left) + len(right) == 8:
#                    t1 = downedges[0].Other(node).Leaves(downedges[0])
#                    pass

#                print("Up", [str(u) for u in up], "Left", [str(l) for l in left], "Right", [str(r) for r in right])

                for up1 in range(len(up) - 1):
                    u1 = up[up1]

                    for u2 in up[up1 + 1:]:

                        # If the two taxa incident to the upward edge are in the 
                        # cover list, we've already emitted any quartets containing
                        # those two taxa

#                        print("Up", str(u1), str(u2))

                        for cv in self.covers:
                            if u1 in cv and u2 in cv:
                                break
                        else:
                            
                            # Not in the cover list, emit all the quartets
                            # for the downward edges

                            rhs = (u1, u2)
                            
                            for l in left:
                                for r in right:

                                    self.nQuartets += 1
     
                                    # Make sure the quartet's taxa are 'sorted'

                                    lhs = (l, r) if l.label <= r.label else (r, l)
                                    yield (lhs, rhs) if lhs[0].label <= rhs[0].label  else (rhs, lhs)
                                
                # Update the list of covers with the current left and right sides

                covered = set(left + right)
                newcovers = []
                for cover in self.covers:
                    if not cover <= covered:
                        newcovers.append(cover)

                self.covers = newcovers + [covered]

#                pp(self.covers)
 
# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
