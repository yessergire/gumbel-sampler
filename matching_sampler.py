import numpy as np
import numpy.random as npr

import networkx as nx
from networkx.algorithms import bipartite, matching

import scipy as sp
import scipy.optimize

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


# Assumptions:
# 1. Each call fixes previous j matches
# 2. We consider only the next j+1 matches
#    that is we perturb (N-j-1)x(N) entries
def partial_upperbound_on_logZ(biadjacency, samples, j):
    # j - current row
    # sample containing j+1 items
    # perturb and map M-times then average

    N = biadjacency.shape[0]

    # Copy original biadjacency matrix
    weights = biadjacency.toarray()+.0

    # Set samples matches
    print(samples)
    for i, x in enumerate(samples):
        w = weights[i,x]
        weights[i] = 0
        weights[i,x] = w

    # Force the matching to be perfect
    weights[weights==0.0] = -np.inf

    # For now assume M = 1
    M = 10
    psi = np.zeros(M)
    for i in range(M):

        # Generate iid Gumbel noise
        noise = GumbelNoise((N-j, N))

        # Ignore already sampled vertices
        #weights[:, samples] = 0#-np.inf
        #noise[:, samples] = 0

        #print(weights)
        # Add noise to weights
        weights[j:] += np.exp(noise)

        # MAP
        left, right = sp.optimize.linear_sum_assignment(weights, maximize=True)
        #print(left)
        #print(right)

        # Store the current MAP value
        psi[i] = np.log(weights[left, right]).sum()

    return psi.mean()


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
    biadjacency = 0.0 + bipartite.biadjacency_matrix(B, row_order=left_partition, column_order=right_partition)

    # Reject-symbol
    n = len(left_partition)
    reject = "R"
    BIPARTITE_SAMPLE_INDECIES = []
    indecies_to_labels = lambda idx: list(map(lambda id: right_partition[id], idx))
    labels_to_indecies = {x:i for i, x in enumerate(right_partition)}

    j = 1
    # This is the 0th psi?
    prev_psi = partial_upperbound_on_logZ(biadjacency, [], 0)
    # sample matches sequentially
    while j < n:
        v = left_partition[j-1]
        psi = {}

        # Probability of (v,u) given v
        # The nth index contains p(reject)
        P = np.zeros(len(B[v])+1)

        # We consider adjecent nodes of v in sequence.
        # At each iteration we consider (v,u) to be a
        # match and ignore other nodes adjacent to v.
        for u in B[v]:
            if labels_to_indecies[u] in BIPARTITE_SAMPLE_INDECIES:
                continue

            # generate upper bounds of logZ
            tmp_sample = BIPARTITE_SAMPLE_INDECIES + [labels_to_indecies[u]]
            psi[u] = partial_upperbound_on_logZ(biadjacency, tmp_sample, j)

            # calculate probability of (v,u)
            k = labels_to_indecies[u]
            P[k] = np.exp(psi[u])/ np.exp(prev_psi)
        #prev_psi = np.mean(list(psi.values())) # NOT CORRECT!
        #print(P)
        #print(psi)
        #print(prev_psi)

        # Probability of rejection
        P[-1] = 1 - np.sum(P)

        # Tmp distribution
        # Makes sure that sampling is done without replacement
        # P = tmp_dist(n, j-1, BIPARTITE_SAMPLE_INDECIES, 0.1)

        # sample from [j] U {'R'}

        sampleId = npr.choice(list(B[v])+[reject], p=P)
        if sampleId == reject:
            print("Sampler restarted!")
            j = 1
            prev_psi = partial_upperbound_on_logZ(biadjacency, [], 0)
            biadjacency = 0.0 + bipartite.biadjacency_matrix(B, row_order=left_partition)
            BIPARTITE_SAMPLE_INDECIES = []
        else:
            BIPARTITE_SAMPLE_INDECIES.append(labels_to_indecies[sampleId])
            #biadjacency[j-1] = 0
            #biadjacency[j-1, sampleId] = 1
            j += 1

    # map ids to node labels
    BIPARTITE_SAMPLE = map_indecies_to_labels(BIPARTITE_SAMPLE_INDECIES)

    # add remaining node
    BIPARTITE_SAMPLE.append(list(set(right_partition) - set(BIPARTITE_SAMPLE))[0])

    return BIPARTITE_SAMPLE
