"""
Generate a set of k-tets from a set of quartets.

The quartets need to be known to be compatable
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ["KTetBuilder", "KTet"]

# **************************************************
# Copyright (c) 2013 Ralph W. Crosby, Texas A&M University, College Station, Texas
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# **************************************************

from   operator import attrgetter
import random
import sys
from   time     import perf_counter

from TreeUtilities.Quartets import Quartets

# *************************************************************************

class KTet(object):
    """
    Class representing a split with a arbitrary, non-overlapping sets of 
    taxa on each side of the split.
    """
    
    txsort = lambda s : sorted(s, key=attrgetter('label'))
    
    # *****************************************************************
    
    def __init__(self, ktet=None):
        if ktet:
            self.left = set(ktet[0])
            self.right = set(ktet[1])
        else:
            self.left = set()
            self.right = set()

    # *****************************************************************

    def __str__(self):
        """Return a formatted string of the k-tet"""
        
        return "{}|{}".format(','.join([str(t) for t in KTet.txsort(self.left)]),
                              ','.join([str(t) for t in KTet.txsort(self.right)]))

    # *****************************************************************

    def __len__(self):
        """Return the number of taxa in the k-tet"""
        return len(self.left) + len(self.right)

    # *****************************************************************

    def __call__(self):
        """Return the k-tet as a tuple of (sorted) lists"""

        return (KTet.txsort(self.left),
                KTet.txsort(self.right))

    # *****************************************************************

    def _AddQuartet(self, quartet):
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
                        
    # *****************************************************************

    _compatabilitylist = {'l r ' : 2, 'l  r' : 2,
                          ' lr ' : 2, ' l r' : 2,
                          'r l ' : 2, 'r  l' : 2,
                          ' rl ' : 2, ' r l' : 2,
                          'llr ' : 3, 'll r' : 3,
                          'rrl ' : 3, 'rr l' : 3,
                          'r ll' : 3, ' rll' : 3,
                          'l rr' : 3, ' lrr' : 3,
                          'llrr' : 4, 'rrll' : 4,
                           }
    
    def _QuartetCompatability(self, quartet):
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
            return  KTet._compatabilitylist[qcompat]
        except KeyError:
            return  0

# *************************************************************************

class KTetBuilder(list):
    """
    Helper class to build a set of k-tets.
    
    The actual K-tets are just represented as a list of tuples with the
    two elements of the tuple representing the left and right sides of the k-tet.

    This class provides an interable across the set of k-tets generated.
    """

    # *****************************************************************

    ADD_LARGEST = 1
    ADD_SMALLEST = 2
    ADD_RANDOM = 3

    TWO_COMPAT = True
    TWO_NOT_COMPAT = False
    
    # *****************************************************************
    
    def __init__(self, 
                 source, 
                 addpolicy = ADD_RANDOM,
                 twopolicy = TWO_COMPAT,
                 verbose   = True,
                 debug     = False):
        """
        Source defines an iterable that will provide the quartets.
        In this context it could be an unrooted tree or a set returned
        from the mongodb.

        In the fullness of time both serial and parallel versions will
        be provided.
        """

        self.addpolicy = addpolicy
        self.twopolicy = twopolicy
        self.verbose   = verbose
        self.debug     = debug

        self._Build(source)

    # *****************************************************************

    def __iter__(self):
        return super().__iter__()

    # *****************************************************************

    @property
    def stats(self):
        """
        Return the collection of statistics
        """
        return [("Total Quartets Processed", self.nq),
                ("KTets Produced", len(self)),
                ("Quartets with 0 or 1 Compatable Taxa", self.compat[0]),
                ("Quartets with 2 Compatable Taxa", self.compat[2]),
                ("Quartets with 3 Compatable Taxa", self.compat[3]),
                ("Quartets with 4 Compatable Taxa", self.compat[4]),
                ("Largest set of 3 Taxa Compatable Ktets", self.max3list),
                ("Largest set of 2 Taxa Compatable Ktets", self.max2list),
                ("Refining quartets", self.refiningq),
                ("Ambiguous quartets", self.ambiguousq),
                ("Compatability Check Time", self.compatTime),
                ("KTet Update Time", self.updateTime),
                ("Add Policy", "Add to Largest" if self.addpolicy == KTetBuilder.ADD_LARGEST else \
                               "Add to Smallest" if self.addpolicy == KTetBuilder.ADD_SMALLEST else \
                               "Add to Random"),
                ("Use Two Compat Quartets", "Yes" if self.twopolicy else "No"),
               ]

    # *****************************************************************

    def _Build(self, source):
        """
        Build the k-tets from the iterable
        """

        self.compat = [0,0,0,0,0]                 # Counts by level of compatability
        self.max3list = 0                         # Largest set of 3 compat ktets
        self.max2list = 0                         # Largest set of 2 compat ktets
        self.refiningq = 0                        # Quartets with only 1 compatable ktet
        self.ambiguousq = 0                       # Quartets with multiple compatable ktets

        self.nq = 0                               # Number of quartets processed

        self.compatTime = 0.0                     # Total time to check compability 
        self.updateTime = 0.0                     # Total time to update k-tets

        for q in source:

            if self.verbose and self.nq % 100000 == 0:
                print("{:,d} quartets processed, {:,d} KTets produced".format(self.nq, len(self)), file=sys.stderr)

            sttm = perf_counter()
            updateTime = 0.0

            store2 = self.twopolicy
            compkt = []
            for kt in self:

                clvl = kt._QuartetCompatability(q)
                if clvl == 4:
                    self.compat[4] += 1
                    break
                if clvl == 3:
                    if store2:
                        store2 = False
                        compkt = []
                    compkt.append(kt)
                elif clvl == 2 and store2:
                    compkt.append(kt)

            else:

                ectm = perf_counter()

                if self.debug:
                    print("{:9,d}: {},{}|{},{}".format(self.nq, q[0][0].label, q[0][1].label, q[1][0].label, q[1][1].label ), file=sys.stderr)
                    print("           List:", len(compkt), store2, file=sys.stderr)
                    for pkt in compkt:
                        print("                ", pkt, file=sys.stderr)
            
                if len(compkt) == 0:

                    self.append(KTet(q))
                    self.compat[0] += 1

                elif len(compkt) == 1:

                    compkt[0]._AddQuartet(q)

                    if store2:
                        self.compat[2] += 1
                        self.max2list = max(1, self.max2list)
                    else:
                        self.compat[3] += 1
                        self.max3list = max(1, self.max3list)
                    self.refiningq += 1

                else:

                    if self.addpolicy == KTetBuilder.ADD_LARGEST:
                        kt = max(compkt, key=len)
                    elif self.addpolicy == KTetBuilder.ADD_SMALLEST:
                        kt = min(compkt, key=len)
                    else:
                        kt = random.choice(compkt)

                    kt._AddQuartet(q)

                    if store2:
                        self.compat[2] += 1
                        self.max2list = max(len(compkt), self.max2list)
                    else:
                        self.compat[3] += 1
                        self.max3list = max(len(compkt), self.max3list)
                    self.ambiguousq += 1

                updateTime = perf_counter() - ectm

            self.nq += 1
            self.updateTime += updateTime
            self.compatTime += perf_counter() - sttm - updateTime

# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
