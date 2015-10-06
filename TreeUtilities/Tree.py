"""
Definitions for the tree objects

Each node has a parent pointer generated using the Node super class

Modeled after Thomas Mailund's tree in his newick parser.
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ["Tree", "Edge", "Inner", "Leaf", "TreeVisitor","DAG","DAGNode"]

# *************************************************************************

from   pympler.asizeof import asizeof
import scipy.special
import sys
import asyncio

# *************************************************************************

class Tree(object):
    """
    Container class for trees.
    Mostly contains the lists of edges and vertices
    """

    def __init__(self):
        self.edges        = []                    # List of edges
        self.vertices     = []                    # List of leafs
        self.nodes        = []                    # Dictionary indexed by node id
        self._diameter    = 0                     # Tree diameter, 0 indicates not set
        self._cacheLeaves = False                 # Don't initially allow caching leaf sets
        self._nLeaves     = 0                     # Number of leaves in the current tree
        self._totLeafLen  = 0                     # Total taxa label length
        self._root        = None                  # Root of the tree



    @property
    def diameter(self):

        if not self.binary:
            raise ValueError("Tree not binary");

        if self._diameter:
            return self._diameter

        self._diameter = _Diameter(self.root if self.root else self.vertices[0])
        return self._diameter

    @property
    def leaves(self):
        """Return list of leaves in the tree"""
        return [l for l in self.vertices if type(l) is Leaf]

    @property
    def leafSet(self):
        """Return the names of the leaves in the tree as a set"""
        return set(str(l) for l in self.vertices if type(l) is Leaf)

    @property
    def memory(self):
        """Return memory utilization for the tree"""
        return asizeof(self)

    @property
    def maxDegree(self):
        """Return the degree of the tree"""
        return max(len(v.edges) for v in self.nodes)

    @property
    def binary(self):
        """Return true if the tree is properly binary"""
        return self.maxDegree == 3

    @property
    def nQuartets(self):
        """Return the number of quartets in the tree"""
        return int(scipy.special.binom(len(self.leaves), 4))

    @property
    def cacheLeaves(self):
        """Return the leaf cache statue"""
        return self._cacheLeaves

    @property
    def nLeaves(self):
        """Return number of leaf nodes in tree"""
        return self._nLeaves

    @property
    def meanLeafLen(self):
        """Return average leaf label length"""
        return self._totLeafLen / self._nLeaves

    @cacheLeaves.setter
    def cacheLeaves(self, v):
        """Set the leaf cache statue"""

        # If the cache is being turned off, clear the cache
        if self._cacheLeaves and not v:
            for v in self.vertices:
                if type(v) == Inner:
                    v.leaves = []

        self._cacheLeaves = v

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, node):
        self._root = node

        # Set the parent pointers
        
        class Visitor(TreeVisitor):

            def PreTreeVisit(self, t, inedge):
                t.parent = inedge

            def VisitLeaf(self, l, inedge):
                l.parent = inedge
                
        node.DFS(Visitor(), None)

    def set_children(self): # Sets children for each tree node
        all_nodes = []
        root = [self._root, None] # root and no incoming edge
        all_nodes.append(root)
        ctr = 0
        while(all_nodes):
            ctr += 1
            top = all_nodes.pop()
            if not(type(top[0]) is Leaf and (not top[0] is self._root) ):
                for e in top[0].edges:
                    if not e is top[1]:
                        next = [e.Other(top[0]), e]
                        all_nodes.append(next)
                        top[0].AddChild(e.Other(top[0]))

    def Add(self, o):
        """Add an item to the tree, does not adjust tree!"""
        if type(o) is Edge:
            self.edges.append(o)
        else:
            self.vertices.append(o)
            if type(o) is Leaf:
                self._nLeaves += 1
                self._totLeafLen += len(o.label)
        self._diameter = 0

    def Remove(self, o):
        """Remove an item from the collection, does not adjust tree!"""
        if type(o) is Edge:
            self.edges.remove(o)

        else:
            self.vertices.remove(o)
            if type(o) is Leaf:
                self._nLeaves -= 1
                self._totLeafLen -= len(o.label)
        self._diameter = 0

    def DFS(self, visitor):
        """Depth first search of the tree starting at the root"""

        if not self._root:
            _unrooted_dfs(visitor)
        else:
            [e.Other(self._root).DFS(visitor, e) for e in self._root.edges]

    def Find(self, item):
       if type(item) == str:
            return next(v for v in self.vertices if v.label == item)

    def MRCA (self, leaves):

        if not self.root:
            raise exception.RuntimeError("Cannot obtain MRCA on unrooted tree")

        path = self._make_path_to_root(leaves[0])

        for leaf in leaves[1:]:
            path = self._find_common_path(path, leaf)

        return path[-1]

    def Newick(self, n, incoming):
        if incoming.length:
            bl = ":" + str(incoming.length)
        else:
            bl = ""

        if type(n) is Leaf:
            return str(n) + \
                n.AttrStr() + \
                bl
        else:
            return '(' + \
                ','.join([self.Newick(e.Other(n), e) for e in n.edges if not e is incoming]) + \
                ')' + \
                n.AttrStr() + \
                bl

    def __str__(self):
        """Return Newick representation of the tree"""
        if self._root:
            newicks = (self.Newick(e.Other(self._root), e) for e in self._root.edges)
        else:
            newicks = (self.Newick(n, self.edges[0]) for n in self.edges[0].nodes)

        return "({});".format(','.join(newicks))

    def __repr__(self):
        """Return a string representation of the whole tree"""

        return '\n'.join(repr(v) for v in self.vertices) + '\n' + '\n'.join(repr(e) for e in self.edges)

    def _find_common_path(self, path1, leaf):
        path2 = self._make_path_to_root(leaf)
        for i in range( min( len(path1),len(path2) ) ):
            if not path1[i] is path2[i]:
                break
        else:
            i += 1

        return path1[0:i]

    def _make_path_to_root(self, leaf):
        node = self.Find(leaf)
        path = []

        edge = node.parent
        while True:
            node = edge.Other(node)
            path.append(node)
            if node is self.root:
                break
            edge = node.parent

        return list(reversed(path))

    def _unrooted_dfs(self,
                      visitor,
                      edge=None):
        """Depth first search of the tree optionally starting with a specific edge"""

        if not self.edges:      # Handle empty tree
            return

        if not edge:
            edge = self.edges[0]

        [n.DFS(visitor, edge) for n in edge.nodes]

# *************************************************************************

class Edge(object):
    """
    Representation of a edge on the tree.
    Contains parent and child pointers as well as the branch length
    """

    def __init__(self,
                 nodes=[],                        # Connected nodes
                 tree=None,                       # Parent tree
                 length=0):                       # Edge length (weight)

        self.length = length

        self.nodes  = []
        [self.AddNode(n) for n in nodes]

        if tree:
            tree.Add(self)

    def __repr__(self):
        return "E[{},{},{}]".format(self.nodes[0].Id() if len(self.nodes) > 0 else "None",
                                    self.length,
                                    self.nodes[1].Id() if len(self.nodes) > 1 else "None")

    def AddNode(self, node):
        """
        Add the connection to a node connecting the node to the edge as well
        Returns the node added.
        """
        self.nodes.append(node)
        if self not in set(node.edges):
            node.edges.append(self)

        return node

    def RemoveNode(self, node):
        """
        Remove the connection to a node disconnecting the edge from the node as well
        Returns the node removed
        """
        self.nodes = [n for n in self.nodes if not n is node]
        if self in set(node.edges):
            node.edges = list(set(node.edges) - set([self]))
             
        return node

    def Depth(self, node):
        """
        Return the depth of the tree pointing downward from the
        specified node
        """

        depth = _DepthVisitor()
        self.Other(node).DFS(depth, self)
        return depth.depth

    def Other(self, node):
        """Return reference to the other end of the edge from node"""

        ni = self.nodes.index(node)

        try:
            other = self.nodes[not ni]
        except IndexError:
            other = None

        return other

# *************************************************************************

class Node(object):
    """
    Class with a edge list and node id.
    The node id is not the same as a taxa label,
    it's just an id for this particular node.
    """

    nodeid = 0

    def __init__(self,
                 edges=[],                        # List of edges to connect
                 label=None,                      # Label for the node
                 tree=None,                       # Parent tree
                 attrs=[],                        # Attributes for the node
                 parent=None,                     # Edge pointing to parent (rooted only) 
                 _children=[],                    # children (nodes) of the current node
                 txId='',                         # the txId of the node (minimum of the children's txId), or leaf's label if node is leaf  
                 _key=''):                        # the key of the node

        self.edges = []
        [self.AddEdge(e) for e in edges]

        self.label  = label
        self.nodeid = Node.nodeid
        self.attrs  = attrs
        self.parent = parent                  # not maintained on tree editing      

        self._children = []
        [self.AddChild(n) for n in _children]

        self.parent = parent

        self.txId = txId

        self._key = _key
                                                  

        if tree:
            tree.Add(self)
            tree.nodes.append(self)
            self.nodeid = len(tree.nodes) - 1     # nodeid is index into trees node list
            self.tree = tree
        else:
            global nodeid
            self.nodeid = Node.nodeid
            Node.nodeid += 1

    def __repr__(self):
        """Print label and nodeid"""
        return "{},{},{}".format(self.Id(), ','.join([repr(e) for e in self.edges]), self.AttrStr())

    def Id(self):
        """Return the repr"""
        return "N{}-'{}'".format(self.nodeid, self.label if self.label else "")

    def AddEdge(self, edge):
        """
        Add an edge setting the pointer in the edge object
        Returns the edge added
        """
        self.edges.append(edge)
        if self not in set(edge.nodes):
            edge.nodes.append(self)

        return edge

    def AttrStr(self):
        """Return any attributes in string format"""

        if not self.attrs:
            return ""

        attrs = []
        for v in self.attrs:
            if type(v) is list:
                attrs.append((v[0], '{' + ','.join(v[1])+ '}'))
            else:
                attrs.append(v)

        return "[&" + ','.join("{}={}".format(a[0],a[1]) for a in attrs) + "]"

    def RemoveEdge(self, edge):
        """
        Remove an edge cleaning up the pointer in the edge object
        Returns the edge removed
        """
        self.edges.remove(edge)
        if self in set(edge.nodes):
            edge.nodes.remove(self)

        return edge

    def AddChild(self, node):
        self._children.append(node)

    def __lt__(self, other):
            return self.txId < other.txId

    def MakeKey(self): #creates the key for the children, through concatenating the keys of its children.

        self = self._children
        self = sorted(self)
    
        key = '('
        ctr = 0
        for child in self:
            ctr+=1
            if(type(child) is Leaf):
                key+= child.label
            else:
                key+=child._key
            if not(ctr == len(self) - 1):
                key+=','

        key+=')'

        return key


# *************************************************************************

class Leaf(Node):
    """
    Leaf just has a label in addition to the base node class attributes
    """

    def __init__(self,
                 label= None,
                 edge = None,                      # One outgoing edge only
                 tree = None,                      # Ref to base unrooted tree
                 attrs = [],                      # List of node attributes
                 parent=None,                     # Edge pointing to parent (rooted only) 
                 _children=[],                    # children (nodes) of the current node
                 txId='',                         # the txId of the node (minimum of the children's txId), or leaf's label if node is leaf  
                 _key=''):                        # the key of the node

        super().__init__(edges=[edge] if edge else [],
                         label=label,
                         tree=tree,
                         attrs=attrs)

        self.parent = parent                  # not maintained on tree editing      

        self._children = []
        [self.AddChild(n) for n in _children]

        self.parent = parent

        self.txId = txId

        self._key = _key

    def __str__(self):
        """Just print the label"""
        return self.label

    def __repr__(self):
        """Generate debugging string for the node"""
        return "L[{}]".format(super().__repr__())

    def __lt__(self, other):
            return self.txId < other.txId

    def Leaves(self, incoming):
        """Just return this node as a list"""
        return [self]

    def DFS(self, visitor, incoming):
        """Just call the leaf visit method"""
        visitor.VisitLeaf(self, incoming)

# *************************************************************************

class Inner(Node):
    """
    Define an inner node with a a set of edges
    """

    def __init__(self,
                 label=None,
                 edges=[],
                 tree=None,                          # Ref to base unrooted tree
                 attrs=[],
                 parent=None,                     # Edge pointing to parent (rooted only) 
                 _children=[],                    # children (nodes) of the current node
                 txId='',                         # the txId of the node (minimum of the children's txId), or leaf's label if node is leaf  
                 _key=''):                        # the key of the node

        super().__init__(edges=edges, label=label, tree=tree, attrs=attrs)

        self.leaves = []

        self.parent = parent                  # not maintained on tree editing      

        self._children = []
        [self.AddChild(n) for n in _children]

        self.parent = parent

        self._key = _key

    def __str__(self):
        """Just output the nodeid"""
        return "N{}".format(self.nodeid)

    def __repr__(self):
        """Generate debugging string for the node"""
        return "I[{}]".format(super().__repr__())

    def Leaves(self, incoming):
        """Return list of leaves in a subtree"""

        if not self.tree.cacheLeaves:
            self.leaves = []

        if not self.leaves:
            edges = self.Other(incoming)
            self.leaves = []
            for e in edges:
                self.leaves.extend(e.Other(self).Leaves(e))

        return self.leaves

    def Other(self, incoming):
        """Return the other edges associated with a node"""
        return [e for e in self.edges if not e is incoming]

    def DFS(self, visitor, incoming):
        """
        Perform a depth first search of the subtree
        considering incoming as the parent pointer
        """

        visitor.PreTreeVisit(self, incoming)
        [e.Other(self).DFS(visitor, e) for e in self.edges if not e is incoming]
        visitor.PostTreeVisit(self, incoming)

# *************************************************************************

class TreeVisitor(object):
    """Visitor base class"""

    def PreTreeVisit(self, t, incoming=None):
        """Called before exploring a subtree"""
        pass

    def PostTreeVisit(self, t, incoming=None):
        """Called after exploring a subtree"""
        pass

    def VisitLeaf(self, l, incoming=None):
        """Called when visiting a leaf node"""
        pass

# *************************************************************************

class _DepthVisitor(TreeVisitor):
    """
    Depth first search visitor class to find the depth of the tree
    """

    def __init__(self):
        self.depth = 0;
        self.__curdepth = 0

    def PreTreeVisit(self, t, incoming=None):
        self.__curdepth += 1

    def PostTreeVisit(self, t, incoming=None):
        self.__curdepth -=1

    def VisitLeaf(self, l, incoming=None):
        if self.__curdepth > self.depth:
            self.depth = self.__curdepth

# **************************************************

def _Diameter(leaf):
    """Return the diameter of the tree"""

    if not leaf or not len(leaf.edges):
        return 0

    diameter, height = _Diameter2(leaf.edges[0].Other(leaf), leaf.edges[0])
    return diameter

def _Diameter2(node, inedge):
    """
    Compute the subtree diameter,
    inedge is the incomming edges from the parent
    """

    if type(node) is Leaf:
        return 0, 0

    # Go through the other edges incident to the current node other
    # than the parent

    dh = [(_Diameter2(e.Other(node), e)) for e in node.edges if not e is inedge]

    heights = sorted([v[1] for v in dh], reverse=True)
    diameter = max(heights[0] + heights[1] + 1, max([v[0] for v in dh]))

    return diameter, heights[0] + 1

# *************************************************************************
class DAGNode(object):
    '''
    Represents each node of the DAG.
    '''
    def __init__(self,
                 dag, # the DAG that the node is contained in 
                 in_edges=0, # in edges of the node
                 out_edges=[], # out edges of the node
                 nodeID=-1,    # ID of the node
                 node_label = '', # the label of the node
                 shared = False, # is False if the DAGNode only appears in one tree, True otherwise 
                 tree_shared = 1, # number of trees in the DAG that contain the DAGNode
                 my_top_nodes = []): # holds the top nodes that the node is under (can also be used to determine which trees the node is part of)

        #self.lock = asyncio.Lock() # the lock that will be used in multiprocessing

        self.nodeID = nodeID

        self.in_edges = 0

        self.out_edges = []
        [self.AddEdge(e) for e in out_edges]

        self.dag = dag

        self.visited = False

        # the edges are directed, the first component is the node the edge emanates from, and the components are the nodeID's

        self.leaves = []  

        self.node_label = node_label

        self.tree_shared = tree_shared

        self.shared = shared

        self.my_top_nodes = my_top_nodes

    def __repr__(self):
        return str(self.node_label)

    
    def Leaves(self): # outputs the leaves of the DAG that are under the current DAGNode
        if len(self.out_edges) == 0:
            self.leaves = [self]

        else:
            self.leaves = []
            for edge in self.out_edges:
                self.leaves.extend(self.dag.DAGNodes[edge[1]].Leaves())

        return self.leaves

    def AddEdge(self, edge):
        # each edge in DAG is represented by two numbers, each of which are node IDs
        if edge[0] == self.nodeID:
            self.out_edges.append(edge)
        elif edge[1] == self.nodeID:
            self.in_edges += 1


    def __lt__(self, other):
        return self.node_label < other.node_label # we will only use this sorting method for sorting the taxa nodes



class DAG(object):
    """
    Represents the DAG. The repr function prints it out in Graphviz format.
    """

    def __init__(self,
                 nodes=[], # contains the nodes' enumeration
                 edges=[], # contains the edges
                 node_labels=[], # contains what each node will be labeled with in its Graphviz representation 
                 numTrees = 0, # the number of trees that were combined to form the DAG
                 DAGNodes = [], # the DAGNode nodes in the DAG
                 shared_nodes = [], # holds whether each node is shared or not shared
                 tree_shared_nodes = []): # holds how many trees share each node

        
        self.nodes = []
        [self.AddNode(n) for n in nodes]

        self.edges = []
        [self.AddEdge(e) for e in edges]

        self.node_labels = node_labels

        self.numTrees = numTrees

        self.DAGNodes = []
        [self.AddDAGNode(n) for n in DAGNodes]

        self.leaves = [] # call the AddLeaves function to add the leaves to the DAG

        self.shared_nodes = shared_nodes

        self.tree_shared_nodes = tree_shared_nodes

    def __repr__(self):   
        myStr = "" 
        myStr += "digraph G {"
        myStr += "\n"
        for edge in self.edges:
            myStr += str(self.node_labels[edge[0]])
            myStr += "->"
            myStr += str(self.node_labels[edge[1]])
            myStr += ";"
            myStr += "\n"
        
        myStr+="}"
        return myStr

    def AddNode(self, node):
        self.nodes.append(node)

    def RemoveNode(self, node):
        self.nodes.remove(node)
    
    def AddEdge(self, edge):
        self.edges.append(edge)

    def RemoveEdge(self, edge):
        self.edges.remove(edge)

    def NumNodes(self): # returns the number of nodes in the DAG
        return len(self.nodes)

    def AddDAGNode(self, myDAGNode):
        self.DAGNodes.append(myDAGNode)

    def RemoveDAGNode(self, myDAGNode):
        self.DAGNodes.remove(myDAGNode)

    def AddLeaves(self):
        for n in self.DAGNodes:
            if (not n in self.leaves) and (len(n.out_edges) == 0):
                self.leaves.append(n)

    def CreateDAGNodes(self):
        for n in range(0,len(self.nodes)):
            myDAGNode = DAGNode(self,[],[],n,self.node_labels[n],self.shared_nodes[n],self.tree_shared_nodes[n],[]) # will be the nth node in DAGNodes
            
            self.AddDAGNode(myDAGNode)

        for e in self.edges:
            self.DAGNodes[e[0]].AddEdge(e)
            self.DAGNodes[e[1]].AddEdge(e)

        self.AddTopNodes()

    def AddTopNodes(self):
        for t in self.top():
            stack = [t]
            while(stack):
                top = stack.pop()
                top.my_top_nodes.append(t)
                for edge in top.out_edges:
                    stack.append(self.DAGNodes[edge[1]])

    def DAGNodesSize(self):
        '''
        number of nodes in DAG
        '''
        return len(self.DAGNodes)

    def top(self):
        top_nodes = []
        for dagnode in self.DAGNodes:
            if dagnode.in_edges == 0:
                top_nodes.append(dagnode)
        return top_nodes

    def topGenerator(self):
        for dagnode in self.DAGNodes:
            if dagnode.in_edges == 0:
                yield dagnode

    def TreeFormedNodes(self, node):
        all_nodes = []
        for mynode in self.DAGNodes:
            if mynode.my_top_nodes[0] == node:
                all_nodes.append(mynode)

        return all_nodes

    def Combine(self, treelist): # treelist is a list of rooted trees, and we combine them to make the DAG.

        all_nodes = [] # is the stack of tree nodes
        node_dictionary = {} # is the dictionary holding node as key, and the number of its children we have visited
        hash_dictionary = {} # is the dictionary holding the hashes for each node
        all_created_nodes = {} # all created nodes in the DAG, is indexed by nodes. It is possible that two nodes point to same number.
        DAG_Node = 0 # number of nodes we have created in DAG
        all_created_edges = [] # all of the edges in the DAG, represented as an directed edge, where the edge emanates from the first component
        node_label = [] # what the nodes of the representation of the DAG will be labeled with
        treelist_size = 0 # size of treelist
        shared = {} # holds False if the corresponding node is not shared, otherwise True
        tree_shared = {} # holds how many trees share the corresponding node

        for tree in treelist: 
            treelist_size += 1
            all_nodes.append(tree.root)
            for node in tree.nodes:
                node_dictionary[node] = 0 
    

        while(all_nodes):
            top = all_nodes[len(all_nodes)-1]
            if(type(top) is Inner or top.tree.root == top): #top is an inner node or vertex
               if(node_dictionary[top] == len(top._children)): #if we have already processed the edges from the vertex

                       if(top.tree.root == top): #top is the root of the tree

                           all_created_nodes[top] = DAG_Node #create new node for the root of this tree
                           shared[all_created_nodes[top]] = False
                           tree_shared[all_created_nodes[top]] = 1                        
                           DAG_Node += 1
                           nodeId = all_created_nodes[top]
                           node_label.append(all_created_nodes[top]) # we append the index of the top node

                           for child in top._children: # make the node point to its children in the DAG
                                childId = all_created_nodes[child]
                                edge = (nodeId,childId)
                                all_created_edges.append(edge)

                       else:
                            key = top.MakeKey()
                            top.txId = min(str(n.txId) for n in top._children)

                            if(key not in hash_dictionary):
                                  all_created_nodes[top] = DAG_Node #create new node, as there is no hash for it
                                  shared[all_created_nodes[top]] = False
                                  tree_shared[all_created_nodes[top]] = 1
                                  DAG_Node += 1
                                  hash_dictionary[key] = top
                                  node_label.append(all_created_nodes[top]) # we append the index of the top node
                            else:
                                shared[all_created_nodes[hash_dictionary[key]]] = True
                                tree_shared[all_created_nodes[hash_dictionary[key]]] += 1

                            node = hash_dictionary[key]
                            nodeId = all_created_nodes[node]
                            all_created_nodes[top] = nodeId #make top point to the node
                            top._key = key # set the key of top equal to the key generated

                            for child in top._children: # make the node point to its children in the DAG
                                childId = all_created_nodes[child]
                                edge = (nodeId,childId)
                                all_created_edges.append(edge)

                            # remove top from all_nodes

                       all_nodes.pop()

               else: 

                   nextChild = top._children[node_dictionary[top]]
                   node_dictionary[top]+=1
                   all_nodes.insert(len(all_nodes),nextChild)    

            elif(type(top) is Leaf): #top is a leaf
                top.txId = str(top.label)
                key = '(' + top.label + ')'
                if(key not in hash_dictionary):
                       all_created_nodes[top] = DAG_Node #create new node, as there is no hash for it
                       shared[all_created_nodes[top]] = False
                       tree_shared[all_created_nodes[top]] = 1
                       DAG_Node += 1
                       hash_dictionary[key] = top
                       node_label.append(top.label) # we append the label of the taxa
                else:
                    shared[all_created_nodes[hash_dictionary[key]]] = True
                    tree_shared[all_created_nodes[hash_dictionary[key]]] += 1


                node = hash_dictionary[key]
                nodeId = all_created_nodes[node]
                all_created_nodes[top] = nodeId #make top point to the node 
                top._key = key # set the key of top equal to the key generated
                all_nodes.pop()

        all_created_edges = sorted(set(all_created_edges))

        self.nodes = range(0,DAG_Node)
        self.edges = all_created_edges
        self.node_labels = node_label
        self.numTrees = treelist_size
        self.shared_nodes = shared
        self.tree_shared_nodes = tree_shared
        
        self.CreateDAGNodes()
        self.AddLeaves()


# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
