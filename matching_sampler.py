import numpy as np
import numpy.random as npr
import networkx as nx
from matching_functions import get_partitions

def GumbelNoise(N):
    return -np.log(-np.log(npr.uniform(size=N)))

def perturb_weights(weights, start=0):
    noise = GumbelNoise(len(weights))
    for i, (k,v) in enumerate(weights.items()):
        weights[k] = noise[i]+v
    return weights

def perturb_graph(B):
    weights = nx.get_edge_attributes(B,'weight')
    G = B.copy()
    nx.set_edge_attributes(G, perturb_weights(weights),'weight')
    return G

# TODO: implement code to estimate the upper bounds!
def partial_upperbound_on_logZ(B, j, u):
    # Ideally sum M MAPs following example in the paper
    # 1. v_edges <- edges of v
    # 2. remove all edges of v except (v,u)
    # 3. run MAP
    # 4. match_weight <- ...
    # 5. restore v_edges to B
    # 6. return average match_weight
    return 0



# TODO: comment this function out, when partial_upperbound_on_logZ works
def tmp_dist(n, j, sample, reject_p=0.5):
    D = np.zeros(n+1)
    D[-1] = reject_p
    d = (1-reject_p)/(n-j)
    for i in range(n):
        if i not in sample:
            D[i]=d
    return D


# TODO: Add documention
def perfect_bipartite_matching_sampler(B):
    # Check if B is a bipartite graph that contains some perfect matching
    # with a call to the hungarian algorithm (we'll call the result as MAP config)

    # Initialize
    left_partition, right_partition = get_partitions(B)
    reject = n = len(left_partition)
    BIPARTITE_SAMPLE_INDECIES = []
    map_indecies_to_labels = lambda idx: list(map(lambda id: right_partition[id], idx))

    j = 0
    prev_psi = 0
    # sample matches sequentially
    while j < n-1:
        v = left_partition[j]
        psi = {}

        # Probability of (v,u) given v
        # The nth index contains p(reject)
        P = np.zeros(n+1)

        # We consider adjecent nodes of v in sequence.
        # At each iteration we consider (v,u) to be
        # a match and ignore other nodes.
        for u in B[v]:
            if u in map_indecies_to_labels(BIPARTITE_SAMPLE_INDECIES):
                continue
            # generate upper bounds of logZ
            psi[u] = partial_upperbound_on_logZ(biadjec, j, u)

            # calculate probability of (v,u)
            # TODO: uncomment, when partial_upperbound_on_logZ works
            # P[u] = np.exp(psi[u])/ np.exp(prev_psi)
        # prev_psi = ?

        # Probability of rejection
        P[reject] = 1 - np.sum(P)

        # TODO: comment, when partial_upperbound_on_logZ works
        # Tmp distribution
        # Makes sure that sampling is done without replacement
        P = tmp_dist(n, j, BIPARTITE_SAMPLE_INDECIES, 0.1)

        sampleId = npr.choice(n+1, p=P)
        if sampleId == reject:
            print("Sampler restarted!")
            j = 0
            prev_psi = 0
            BIPARTITE_SAMPLE_INDECIES = []
        else:
            BIPARTITE_SAMPLE_INDECIES.append(sampleId)
            j += 1

    # map ids to node labels
    BIPARTITE_SAMPLE = map_indecies_to_labels(BIPARTITE_SAMPLE_INDECIES)

    # add remaining node
    BIPARTITE_SAMPLE.append(list(set(right_partition) - set(BIPARTITE_SAMPLE))[0])

    return BIPARTITE_SAMPLE
