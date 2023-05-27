import networkx as nx
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.optimize
from networkx.algorithms import bipartite
from scipy.optimize import linear_sum_assignment

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


def perfect_bipartite_matching_sampler(B):
    # Check if B is a bipartite graph that contains some perfect matching
    # with a call to the hungarian algorithm (we'll call the result as MAP config)

    # Initialize
    left_partition, right_partition = get_partitions(B)
    biadjacency = 0.0 + bipartite.biadjacency_matrix(B, row_order=left_partition, column_order=right_partition)

    # Reject-symbol
    n = len(left_partition)
    reject = "R"
    BIPARTITE_SAMPLE_INDICES = []
    indices_to_labels = lambda idx: list(map(lambda id: right_partition[id], idx))
    labels_to_indices = {x:i for i, x in enumerate(right_partition)}

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
            if labels_to_indices[u] in BIPARTITE_SAMPLE_INDICES:
                continue

            # generate upper bounds of logZ
            tmp_sample = BIPARTITE_SAMPLE_INDICES + [labels_to_indices[u]]
            psi[u] = partial_upperbound_on_logZ(biadjacency, tmp_sample, j)

            # calculate probability of (v,u)
            k = labels_to_indices[u]
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
            BIPARTITE_SAMPLE_INDICES = []
        else:
            BIPARTITE_SAMPLE_INDICES.append(labels_to_indices[sampleId])
            #biadjacency[j-1] = 0
            #biadjacency[j-1, sampleId] = 1
            j += 1

    # map ids to node labels
    BIPARTITE_SAMPLE = map_indices_to_labels(BIPARTITE_SAMPLE_INDICES)

    # add remaining node
    BIPARTITE_SAMPLE.append(list(set(right_partition) - set(BIPARTITE_SAMPLE))[0])

    return BIPARTITE_SAMPLE


#def phi(N, samples): pass

def potential(log_weights, match):
  left, right = match
  return log_weights[left, right].sum()

# partial upper bound on logZ
# given a biadjacency weight matrix and
# a list of samples
# return the partial upper bound on logZ
# calculated according to (Hazan et al. 2019)


def get_matches(weights, j):
    N = weights.shape[0]
    log_weights = np.log(weights)
    noise = GumbelNoise((N, N))
    perturbed_weights = log_weights + noise
    _, right = linear_sum_assignment(perturbed_weights, maximize=True)
    return right


def get_upper_bound(bmatrix, samples):
    weights = bmatrix.copy()

    j = len(samples)
    N = bmatrix.shape[0]
    I = np.arange(N)
    for i in range(j):
        weights[i, I != samples[i]] = 0

    log_weights = np.log(weights)
    can_sample = list(set(range(N)) - set(samples))

    M = 100
    psi = 0
    for i in range(M):
        noise = np.zeros((N,N))
        for v in can_sample:
            noise[j:, v] = GumbelNoise(N-j)
        perturbed_weights = log_weights + noise
        left, right = linear_sum_assignment(perturbed_weights, maximize=True)
        psi += potential(perturbed_weights, (left, right))
    return psi / M


def get_distribution(samples, weights):
    # print("  called get_distribution:")
    N, _ = weights.shape
    P = np.zeros(N+1)
    # samples = [5,3]
    can_sample = list(set(range(N)) - set(samples))

    all_phi = get_upper_bound(weights, samples)
    # print('all_phi=%.2f' % (all_phi))

    for v in can_sample:
        phi_v = get_upper_bound(weights, samples + [v])
        P[v] = np.exp(phi_v - all_phi)
        # print('    phi[%d]=%.2f; P[%d]=%.2f' % (v, phi_v, v, P[v]))

    P[-1] = 1 - P.sum()
    # print('    P[R]=%.2f' % (P[-1]))
    return P

def algo1(bmatrix):
    r = 0
    N, _ = bmatrix.shape
    reject = N
    samples = []
    labels = list(map(lambda i: chr(i+97), range(N))) + ['R']
    while len(samples) < N - 1:
        # print('\nBuild distribution for [%s]' % (labels[len(samples)]))
        P = get_distribution(samples, bmatrix)
        # print('Distribution\n', P)
        if P[-1] < 0:
            sampleId = reject
        else:
            sampleId = npr.choice(range(len(P)), p=P)
        # print('sampled', labels[sampleId])
        if sampleId == reject:
            # print("Sampler restarted!")
            r += 1
            samples = []
        else:
            samples.append(sampleId)

    print("\n\nTotal number of restarts =", r)
    # last = list(set(range(N-1)) - set(samples))[0]
    # samples.append([last])
    return samples

def get_sample_graph(N=5):
    B = nx.Graph()

    top_nodes = list(range(N))
    bottom_nodes = list(range(N, N*2))

    B.add_nodes_from(top_nodes, bipartite=0)
    B.add_nodes_from(bottom_nodes, bipartite=1)

    for i in range(N):
        B.add_edge(i, N+i, weight=1)
        if i > 0:
            B.add_edge(i, N+i-1, weight=2.0)
        if i < N-1:
            B.add_edge(i, N+i+1, weight=1.0)

    return B

def main():
    n = 7
    # B = get_sample_graph(n)

    # Initialize
    # left, right = get_partitions(B)
    # bmatrix = bipartite.biadjacency_matrix(B, row_order=left).toarray()
    # bmatrix = np.array(bmatrix, dtype=np.float64)
    # # print('bmatrix\n', bmatrix)

    weights = npr.randint(1,10, size=(n,n))
    samples = algo1(weights)#[j:,  np.arange(n) != v])
    # remapped_samples = list(map(lambda u: u + j if u >= v else u, samples))
    # real_samples = [3] + remapped_samples
    print('samples', samples)

if __name__ == '__main__':
    main()