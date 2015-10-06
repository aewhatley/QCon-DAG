

__all__ = ["DAGQuartets", "MultiprocessDFS"]

from TreeUtilities.Tree import *
from operator import attrgetter
import asyncio
from CombineTrees2 import *
import time


class DAGQuartets(object):

    def __init__(self, dag = None):

        self.dag = dag # DAG that the Quartets belong to
        self.top_nodes = self.dag.top() # top nodes in the dag
        self.nQuartets = 0 # number of Quartets in the DAG
        self.covers = {} # dictionary that holds which taxa are covered keyed by the starting node
        self.processed_nodes = 0 # the number of nodes in the dag that have been visited

            

    def MultiprocessDFS(self, node): # do the DFS from the top node of the tree in each process
            def LCAPairs(self): # return all of the pairs of taxa whose least common ancestor is the top vertex of the tree
                    all_pairs = {}
                    myDAG = self.dag
                #for node in self.dag.top():
                    stack = []
                    all_nodes = []
                    stack.append(node)
                    all_nodes.append(node)
                    all_pairs[node] = {}

                    while(stack):
                        current = stack.pop()
                        for edge in current.out_edges:
                                stack.append(myDAG.DAGNodes[edge[1]])
                                all_nodes.append(myDAG.DAGNodes[edge[1]])
                    while(all_nodes):
                        top = all_nodes.pop()
                    
                        if len(top.out_edges) == 0:
                            continue
                        else:
                            left = self.dag.DAGNodes[top.out_edges[0][1]].Leaves()
                            right = self.dag.DAGNodes[top.out_edges[1][1]].Leaves()
                            for l in left:
                                for r in right:
                                    lhs = lhs = (l, r) if l.node_label <= r.node_label else (r, l)
                                    all_pairs[node][lhs] = top

                    return all_pairs

            '''def AllDescendants(self):
                all_pairs = {}
                myDAG = self.dag
                stack = []
                all_nodes = []

                #for node in self.dag.top():
                stack.append(node)
                all_nodes.append(node)

                while(stack):
                        current = stack.pop()
                        for edge in current.out_edges:
                                stack.append(myDAG.DAGNodes[edge[1]])
                                all_nodes.append(myDAG.DAGNodes[edge[1]])

                for mynode in all_nodes:
                    if mynode in all_pairs:
                        continue
                    else: # was originally else statement
                        all_pairs[mynode] = {}
                        leaves = mynode.Leaves()
                        for leaf in leaves:
                            all_pairs[mynode][leaf] = True

                return all_pairs'''

            def AllDescendants(self):
                all_pairs = {}
                lca = LCAPairs(self)

                for lhs in lca[node]:
                    mynode = lca[node][lhs]
                    if mynode in all_pairs:
                        continue
                    else:
                        all_pairs[mynode] = {}
                        for leaf in mynode.Leaves():
                            all_pairs[mynode][leaf] = True

                return all_pairs

    ##################################################################################        
            lca = LCAPairs(self)
            descendants = AllDescendants(self)
            myDAG = self.dag
            all_nodes = []
            stack = []
            tree_node_position = [] # holds the index of the DAG Node corresponding to the same position in all_nodes
            stack.append(node)
            all_nodes.append(node)

            while(stack):
                current = stack.pop()
                for edge in current.out_edges:
                   if node == myDAG.DAGNodes[edge[1]].my_top_nodes[0]:
                        stack.append(myDAG.DAGNodes[edge[1]])
                        all_nodes.append(myDAG.DAGNodes[edge[1]])
        
            num_to_visit = len(all_nodes)
            while(all_nodes):
                    top = all_nodes.pop()
                    #print('Visited {} of {}'.format(self.processed_nodes, num_to_visit))
                    if len(top.out_edges) == 0 or len(top.out_edges) == 1: # if it is a leaf
                            self.processed_nodes += 1
                            continue

                    
                    else:   

                            left = myDAG.DAGNodes[top.out_edges[0][1]].Leaves()
                            right = myDAG.DAGNodes[top.out_edges[1][1]].Leaves()
                            up = sorted([l for l in self.dag.leaves if not l in left and not l in right]) # add sorted leaves not in left or right to up

                            #print('{} visited node step 1'.format(self.processed_nodes))
                            # emit the quartets that are counted from two vertices
                            for i in range(0,len(up)-1): 
                                u1 = up[i]
                                for j in range(i+1,len(up)):
                                    u2 = up[j]
                                    rhs = (u1, u2)
                                    total = len(top.my_top_nodes)

                                    # set the Quartets


                                    for l in left:
                                          for r in right:
                                            
                                                self.nQuartets += total
                                                #self.nQuartets += 1
                                                lhs = (l, r) if l.node_label <= r.node_label else (r, l)
                                                yield ((lhs[0].node_label, lhs[1].node_label), (rhs[0].node_label, rhs[1].node_label), total) if lhs[0].node_label <= rhs[0].node_label else ((rhs[0].node_label, rhs[1].node_label), (lhs[0].node_label, lhs[1].node_label), total)

                
                            # emit the quartets that are counted from one vertex. They will be double-counted just like the ones above
                            #print('{} visited node step 2'.format(self.processed_nodes))
                            for l in left:
                                for r in right:
                                    total = len(top.my_top_nodes)
                                    rhs = (l, r) if l.node_label <= r.node_label else (r, l)
                                    down_left = sorted([taxa for taxa in left if not taxa == l])
                                    down_right = sorted([taxa for taxa in right if not taxa == r])

                                    for i in range(0,len(down_left)-1):
                                        dl1 = down_left[i]
                                        for j in range(i+1,len(down_left)):
                                            dl2 = down_left[j]
                                            d_pair = (dl1, dl2) if dl1.node_label <= dl2.node_label else (dl2, dl1)

                                            if not l in descendants[lca[node][d_pair]]: # make sure that dl1, dl2, and l are not in the same subtree
                                                lhs = (dl1, dl2) if dl1.node_label <= dl2.node_label else (dl2, dl1)
                                                self.nQuartets += total
                                           
                                                yield ((lhs[0].node_label, lhs[1].node_label), (rhs[0].node_label, rhs[1].node_label), total) if lhs[0].node_label <= rhs[0].node_label else ((rhs[0].node_label, rhs[1].node_label), (lhs[0].node_label, lhs[1].node_label), total)

                                    for i in range(0,len(down_right)-1):
                                        dr1 = down_right[i]
                                        for j in range(i+1,len(down_right)):
                                            dr2 = down_right[j]
                                        
                                            d_pair = (dr1, dr2) if dr1.node_label <= dr2.node_label else (dr2, dr1)

                                            if not r in descendants[lca[node][d_pair]]: # make sure that dr1, dr2, and r are not in the same subtree
                                                lhs = (dr1, dr2) if dr1.node_label <= dr2.node_label else (dr2, dr1)
                                                self.nQuartets += total
                                            
                                                yield ((lhs[0].node_label, lhs[1].node_label), (rhs[0].node_label, rhs[1].node_label), total) if lhs[0].node_label <= rhs[0].node_label else ((rhs[0].node_label, rhs[1].node_label), (lhs[0].node_label, lhs[1].node_label), total)

                                                

                            #print('{} visited node step 3'.format(self.processed_nodes))
                            self.processed_nodes += 1



if __name__ == '__main__':
    #print('This is not the main file.')

    myDAG = DAG()
    myDAG.Combine(ReadIn(None))
    print(myDAG)
    quartet_dictionary = {}
    allQuartets = DAGQuartets(myDAG)
    for t in myDAG.top():
        ctr = 0
        for q in allQuartets.MultiprocessDFS(t):
                        ctr += 1
        print('CTR:')
        print(ctr)
    print('TOTAL')
    print(allQuartets.nQuartets)
    mySortedDictionary = []
    for q in quartet_dictionary:
        mySortedDictionary.append((q[0],q[1],q[2],q[3], quartet_dictionary[q]))
    mySortedDictionary = sorted(mySortedDictionary)
    for q in mySortedDictionary:
        print(q)
    print(allQuartets.nQuartets)

    



  


         


