# simple commands to get to know how to use the networkx module - from networkx documentation file
# Urszula Golyska

import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()

# adding nodes
G.add_node(1)
G.add_nodes_from([2,3])
G.add_nodes_from([(4,{'color':'red'}), (5,{'color':'green'})])  # node attributes

H=nx.path_graph(10)
G.add_nodes_from(H)
G.add_node(H)  # adds H as one node

# adding edges
G.add_edge(1,2)
e=(2,3)     # tuple
G.add_edge(*e)  # unpacking edge tuple

G.add_edges_from([(1,2),(1,3)])
# e=(2,3,{'weight':3.1415}) # wrong implementation
# G.add_edges_from(e) # ebunch - edge with (weight) attributes

G.add_edges_from(H.edges)
G.clear()   # delete all nodes and edges

G.add_edges_from([(1,2), (1,3)])    # nodes will be created automatically
G.add_node(1)
G.add_edge(1,2)
G.add_node('spam')  # adds one node 'spam'
G.add_nodes_from('spam')    # adds four nodes 's', 'p', 'a', 'm'
G.add_edge(3,'m')

# print(G.number_of_nodes())

# assertion
DG = nx.DiGraph()   # directed graph
DG.add_edge(2,1)    # edge in the order 2,1
DG.add_edge(1,3)
DG.add_edge(2,4)
DG.add_edge(1,2)
# assert list(DG.successors(2))==[1,4]
# assert list(DG.edges) == [(2, 1), (2, 4), (1, 3), (1, 2)]

# examining elements of graph
# print(G.nodes)
# print(G.edges)
# print(G.adj[1]) # adjacencies (neighbours)
# print(G.degree[1]) # number of edges incident to 1

# removing elements
G.remove_node(2)
G.remove_nodes_from('spam')
G.remove_edge(1,3)
# print(G.nodes)

# graph constructors
G.add_edge(1,2)
H=nx.DiGraph(G) # create a directed graph using the connections from G

edgelist = [(0,1),(1,2),(2,3)]
H = nx.Graph(edgelist)
# print(H.edges)

# convert_node_labels_to_integers() # for more traditional labels

# accessing edges and neighbours
G = nx.Graph([(1,2,{'color':'yellow'})])
# print(G[1]) # same as G.adj[1]
# print(G[1][2])
# print(G.edges[1,2])

G.add_edge(1,3)
G[1][3]['color']='blue'
G.edges[1,2]['color']='red'
# print(G.edges[1,2])

FG = nx.Graph() # undirected graphs, sees edges twice!
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
for n,nbrs in FG.adj.items():
    for nbr, eattr in nbrs.items():
        wt=eattr['weight']
        # if wt<0.5 : print(f"({n}, {nbr}, {wt:.3})")

# edge properties
for (u,v,wt) in FG.edges.data('weight'):
    if wt<0.5:
        print(f"({u}, {v}, {wt:.3})")

## adding attributes
G = nx.Graph(day="Friday")  # assigning graph attributes
G.graph['day']='Monday'

G.add_node(1,time='5pm')    # node attributes
G.add_nodes_from([3],time='2pm')
# print(G.nodes[1])
G.nodes[1]['room']=714
# print(G.nodes.data())

G.add_edge(1,2,weight=4.2)    # edge attributes
G.add_edges_from([(3,4),(4,5)], color='red')
G.add_edges_from([(1,2,{'color':'blue'}), (2,3,{'weight':8})])
G[1][2]['weight']=4.7
G.edges[3,4]['weight']=4.2
# print(G.edges[1,2])

# directed graphs
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1,2,0.5),(3,1,0.75)])
# print(DG.out_degree(1,weight='weight'))
# print(DG.degree(1,weight='weight'))
# print(list(DG.successors(1)))
# print(list(DG.neighbors(1)))

# analyzing graphs
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3)])
G.add_node("spam") # adds node "spam"
# print(list(nx.connected_components(G)))
# print(sorted(d for n, d in G.degree()))
# print(nx.clustering(G))

sp = dict(nx.all_pairs_shortest_path(G))
print(sp[3])

# drawing graphs
G = nx.petersen_graph()
subax1 = plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')
subax2 = plt.subplot(122)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')


options = {'node_color': 'black','node_size': 100,'width': 3}
subax1 = plt.subplot(221)
nx.draw_random(G, **options)
subax2 = plt.subplot(222)
nx.draw_circular(G, **options)
subax3 = plt.subplot(223)
nx.draw_spectral(G, **options)
subax4 = plt.subplot(224)
nx.draw_shell(G, nlist=[range(5,10), range(5)], **options)

G = nx.dodecahedral_graph()
shells = [[2, 3, 4, 5, 6], [8, 1, 0, 19, 18, 17, 16, 15, 14, 7], [9, 10, 11, 12,13]]
nx.draw_shell(G, nlist=shells, **options)
plt.show()

nx.draw(G)
plt.savefig("path.png")

# from networkx.drawing.nx_pydot import write_dot
# pos = nx.nx_agraph.graphviz_layout(G)
# nx.draw(G, pos=pos)
# write_dot(G, 'file.dot')