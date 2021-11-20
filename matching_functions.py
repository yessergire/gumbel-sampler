import networkx as nx
from networkx.algorithms import bipartite, matching

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
    
    nx.draw(B, with_labels=True, pos=pos,width=width)
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


def maximum_weight_full_matching(B, top_nodes, bottom_nodes):
    weights = nx.get_edge_attributes(B,'weight')

    max_value = max(weights.values())
    min_weights = {key:(max_value-value) for (key, value) in weights.items()}

    min_B = B.copy()
    nx.set_edge_attributes(min_B, min_weights, 'weight')
    match = bipartite.matching.minimum_weight_full_matching(min_B, top_nodes=top_nodes)
    match = [(v,match[v]) for v in top_nodes if v in match.keys()]
    return match
