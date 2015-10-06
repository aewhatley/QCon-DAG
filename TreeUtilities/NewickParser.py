"""
Basic newick parser implemented as a generator

Each node has a parent pointer generated using the Node super class

Originally copied from Thomas Mailund's newick parser but redeveloped
to support attributes.
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ["Parser", "Reader"]

# *************************************************************************

import pdb
import tpg
import sys

from TreeUtilities.Tree import *

class Parser(tpg.VerboseParser):
    r"""

    separator space	'\s+' ;

    token     Val       '[^,:;=()\{\}\[\]\s]+' ;

    token     Colon      ':' ;
    token     Comma      ',' ;
    token     SemiColon  ';' ;
    token     Equal      '=' ;

    token     AttrStart  '\[&' ;
    token     AttrEnd    '\]' ;

    token     OpenParen  '\(' ;
    token     CloseParen '\)' ;

    token     OpenGroup  '{' ;
    token     CloseGroup '}' ;

    ID/i  -> Val/id                                         $ i = id $ ;
    VAL/v -> Val/id                                         $ v = id $ ;
    KEY/k -> Val/id                                         $ k = id $ ;

    START/tree ->                                           $ self.tree = Tree()
        EDGE/edge SemiColon                                 $ self._finalize_tree(edge)
                                                            $ tree = self.tree
    ;

    EDGE/edge ->
        NODE/node BRLEN/brlen                               $ edge = Edge(nodes=[node],length=brlen, tree=self.tree)
    ;

    NODE/node ->
        LEAF/leaf                                           $ node = leaf
        | INNER/inner                                       $ node = inner
    ;

    BRLEN/brlen ->
         Colon VAL/val                                      $ brlen = float(val)
         | EMPTY                                            $ brlen = 0.0
    ;

    LEAF/leaf ->
         ID/id ATTR/attr                                    $ leaf = Leaf(label=id, attrs=attr, tree=self.tree)
    ;

    INNER/inner ->
         SUBTREE/subtree ATTR/attr                          $ subtree.attrs = attr
                                                            $ inner = subtree
    ;

    ATTR/attr ->
        AttrStart KV/kv                                     $ attr = [kv]
                  ( Comma KV/kv                             $ attr.append(kv)
                  )*
        AttrEnd
        | EMPTY                                             $ attr = []
    ;

    SUBTREE/subtree ->
        OpenParen EDGE/edge                                 $ edges = [edge]
                  ( Comma EDGE/edge                         $ edges.append(edge)
                  )*
        CloseParen                                          $ subtree = Inner(edges=edges, tree=self.tree)
    ;

    KV/kv ->
         KEY/key Equal VALUE/value                          $ kv = (key, value)
    ;

    VALUE/value ->
        VAL/val                                             $ value = val
        | OpenGroup VAL/val                                 $ value = [val]
             ( Comma VAL/val )*                             $ value.append(val)
          CloseGroup
    ;

    EMPTY ->
    ;

    """

    # **************************************************

    def __init__(self, input):

        tpg.VerboseParser.__init__(self)

        self.verbose = 0

        if type(input) is str:
            initer = input.split(';')
        else:
            initer = Reader(input, ';')

        self.newickIter = (l for l in initer if len(l.strip()))

    # **************************************************

    def __iter__(self):
        """
        Return trees
        """

        for l in self.newickIter:
            if len(l):
                yield self.parse('START', l + ';')

    # **************************************************

    def __call__(self):
        return self.__iter__()

    # **************************************************

    def _finalize_tree(self, edge):

        topNode = edge.nodes[0]

        # trim off last edge
        edge.RemoveNode(topNode)
        self.tree.Remove(edge)

        # if the node pointed to from the edge is of degree 2 make it the root

        if len(topNode.edges) == 2:
            self.tree.root = topNode 
              
    def AllTreeList(self):
        all_trees = []
        for l in self.newickIter:
            if len(l):
                all_trees.append(self.parse('START', l + ';'))

        return all_trees
# **************************************************

def Reader(inputFile, delimiter, buffersize=100000):
    """Generator to return newick strings"""

    lines = ['']

    for data in iter(lambda: inputFile.read(buffersize), ''):
        lines = (lines[-1] + data).split(delimiter)
        for line in lines[:-1]:
            yield line

    yield lines[-1]

