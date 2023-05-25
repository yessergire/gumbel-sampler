import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
from scipy.optimize import linear_sum_assignment


def get_partitions(G):
    tops, bottoms = bipartite.sets(G)
    top_nodes = list(sorted(tops))
    bottom_nodes = list(sorted(bottoms))
    return top_nodes, bottom_nodes


def draw_bipartite(B, top_nodes=None, bottom_nodes=None, width=None, match=None, show_weights=True):
    if top_nodes is None or bottom_nodes is None:
        top_nodes, bottom_nodes = get_partitions(B)

    pos = {}
    step = 2.0 / (len(top_nodes)-1)

    for i in range(len(top_nodes)):
        y = 1-step * i
        pos[top_nodes[i]] = [-1, y]
        pos[bottom_nodes[i]] = [1, y]

    # or use pos=nx.drawing.layout.bipartite_layout(B, nodes=top_nodes)

    width = None
    if match is not None:
        width = []
        for (u,v) in list(B.edges()):
            if (u,v) in match or (v,u) in match:
                width.append(3)
            else:
                width.append(1)
    
    nx.draw_networkx(B, with_labels=True, pos=pos,width=width)
    if show_weights:
        labels = nx.get_edge_attributes(B,'weight')
        nx.draw_networkx_edge_labels(B, pos, edge_labels=labels)


def total_cost(B, match):
    weights = nx.get_edge_attributes(B,'weight')
    total = 0
    for (u,v) in list(B.edges()):
        if (u,v) in match or (v,u) in match:
            total += weights[(u,v)]
    return total


def get_biadjacency_matrix(B, top_nodes, maximum=False):
  biadjacency = bipartite.biadjacency_matrix(B, row_order=top_nodes)
  biadjacency_array = biadjacency.toarray()
  inf_indecies = (biadjacency_array == 0)
  if maximum:
    biadjacency_array = biadjacency.max() - biadjacency_array
  biadjacency_array[inf_indecies] = np.inf
  return biadjacency_array


def weight_full_matching(B, top_nodes, bottom_nodes, maximum=False):
    biadjacency = get_biadjacency_matrix(B, top_nodes=top_nodes, maximum=maximum)
    match = linear_sum_assignment(biadjacency)
    match_t = [(top_nodes[match[0][i]], bottom_nodes[match[1][i]])  for i in range(len(match[0]))]
    return match_t

