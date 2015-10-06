from TreeUtilities.NewickParser import Parser
from TreeUtilities.Tree import *
from TreeUtilities.Tree import DAG, Node, Edge
from os import listdir
import asyncio

def ReadIn(args):
   
    rooted_trees = []
    common_root = None
    i = 0

    if args == None: # only used for debugging purposes
       for filename in listdir('Trees'):
            f = open('Trees\\'+ filename, 'r')

            parse = Parser(f)
            
            for t in parse: # convert the unrooted tree from the parser into a rooted tree:
                    for n in t.vertices:
                        if len(n.edges) == 2:
                            t.root = n
                            t.set_children()
                            break

                    #rooted_trees.append(t)
                    yield t

    else: # if it is a file

        for f in args:
            fp = open(f, 'r') if type(f) is str else f
            fp.seek(0)
            for t in Parser(fp):
                for n in t.vertices:
                        if len(n.edges) == 2:
                            t.root = n
                            t.set_children()
                            break

                #rooted_trees.append(t)
                yield t
                
                i += 1

    #return rooted_trees
               
#****************************************************************************
if __name__ == '__main__':
    myDAG = DAG()
    myDAG.Combine(ReadIn(None))
    print(myDAG) # print Graphviz representation of tree