"""
Classes representing a collection of bipartitions

In particular as generated from a set of quartets
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ["Bipartition","Bipartitions"]

# *************************************************************************

from   operator import attrgetter
from   numpy    import  random
from   time     import perf_counter

from   TreeUtilities.Quartets import *
from   TreeUtilities.Tree     import *

# *************************************************************************

class Bipartition(object):

    def __init__(self, ntaxa, quartet):
        self.left = set(quartet[0])
        self.right = set(quartet[1])
        self.ntaxa = ntaxa

    def __str__(self):
        """Return a formatted string of the bipartition"""
        return "{}|{}".format(','.join([str(t) for t in sorted(self.left,  key=attrgetter('label'))]),
                              ','.join([str(t) for t in sorted(self.right, key=attrgetter('label'))]))

    def __int__(self):
        """Return the number of taxa in the bipartition"""
        return len(self.left) + len(self.right)

    @property
    def complete(self):
        return self.ntaxa == len(self.left) + len(self.right)

    # def Sort(self):
    #     """Sort the bipartition by taxa labels"""

    #     self.left.sort(key=attrgetter('label'))
    #     self.right.sort(key=attrgetter('label'))
    #     if self.left[0].label > self.right[0].label:
    #         t = self.left
    #         self.left = self.right
    #         self.right = t

    def AddQuartet(self, quartet):
        """Add leafs from a quartet to the bipartition"""

        for half in quartet:

            addLeaf = None
            compat  = None
            for leaf in half:
                if leaf in self.left:
                    compat = self.left
                elif leaf in self.right:
                    compat = self.right
                else:
                    addLeaf = leaf
            if addLeaf and compat:
                compat.add(addLeaf)

    __compatabilitylist = {'l   ' : 1,
                           'r   ' : 1,
                           ' l  ' : 1,
                           ' r  ' : 1,
                           '  l ' : 1,
                           '  r ' : 1,
                           '   l' : 1,
                           '   r' : 1,
                           'll  ' : 2,
                           '  ll' : 2,
                           'rr  ' : 2,
                           '  rr' : 2,
                           'llr ' : 3,
                           'll r' : 3,
                           'rrl ' : 3,
                           'rr l' : 3,
                           'r ll' : 3,
                           ' rll' : 3,
                           'l rr' : 3,
                           ' lrr' : 3,
                           'llrr' : 4,
                           'rrll' : 4,
                           }

    def QuartetCompatability(self, quartet):
        """
        Determine if a quartet is compatable with the bipartition.

        If compatable, returns the number of taxa compatable (1-4)
        """

        qcompat = ""
        for half in quartet:
            for leaf in half:

                if leaf in self.left:
                    compat = 'l'
                elif leaf in self.right:
                    compat = 'r'
                else:
                    compat = ' '

                qcompat += compat

        try:
            return  self.__compatabilitylist[qcompat]
        except KeyError:
            return  0

# *************************************************************************

class Bipartitions(list):
    """
    Class defining a list of bipartitions generated from a set of quartets
    """

    def __init__(self, tree, qpct=1.0, ntaxa=0, startEdge=None, log=None):
        """
        Create the bipartition list selecting from the quartet list
        """

        self.ntaxa = ntaxa                        # Known number of taxa
        self.newq = 0                             # Quartets that start a new biparition
        self.refiningq = 0                        # Quartets the refine a bipartition
        self.redundantq = 0                       # Redundant quartets
        self.ambiguousq = 0                       # Quartets that might fit multiple bipartitions
        self.qproc = 0                            # Number of quartets processed

        self.compatTime = 0.0
        self.updateTime = 0.0

        nQuartets = tree.nQuartets                # Total quartets to process
        if nQuartets < 1000:                      # Don't output too often
            logpct = nQuartets
        else:
            logpct = int(nQuartets/20)            # When to output a log line

        tree.cacheLeaves = True                   # Allow leaf sets to be cached at the inner nodes
        qIter = Quartets(tree, startEdge)         # Generator object for quartets
        for q in qIter:

#            print(str(q[0][0]),str(q[0][1]),str(q[1][0]),str(q[1][1]))

            if log and qIter.nQuartets % logpct == 0:
                log.info("{:,d} quartets seen, {:,d} processed, {:,d} bipartitions created".format(qIter.nQuartets,
                                                                                                   self.qproc,
                                                                                                   len(self)))

            if random.random() > qpct:            # Only select qpct of the quartets
                continue

            # Pass the list of quartets and find compatable bipartitions

            sttm = perf_counter()

            compbi = []
            for b in self:
                compat = b.QuartetCompatability(q)
                if compat > 2:                    # At least three taxa must be compatable
                    compbi.append((b, compat))

            ectm = perf_counter()

            # Depending on the number of compatable bipartitions...

            if len(compbi) == 0:                  # Not compatable, create new bipartition
                self.append(Bipartition(ntaxa, q))
                self.newq += 1

            elif len(compbi) == 1:                # Compatable with a single bipartition
                if compbi[0][1] == 4:             # Redundant
                    self.redundantq += 1
                else:
                    compbi[0][0].AddQuartet(q)    # Add quartet to the bipartition
                    self.refiningq += 1

            else:                                 # Ambiguous
                self.ambiguousq += 1

            self.qproc += 1
            self.updateTime += perf_counter() - ectm
            self.compatTime += ectm - sttm

        qIter.Check()
        self.qin = qIter.nQuartets
        tree.cacheLeaves = False                  # Disable leaf caching

# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
