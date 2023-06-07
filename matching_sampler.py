from time import time

import networkx as nx
import numpy as np
import numpy.random as npr
from scipy.optimize import linear_sum_assignment


def GumbelNoise(N):
    return -np.log(-np.log(npr.uniform(size=N)))

def get_upper_bound(log_weights):
    M = 50
    psi = 0

    for i in range(M):
        perturbed_weights = log_weights + GumbelNoise(log_weights.shape)
        left, right = linear_sum_assignment(perturbed_weights, maximize=True)
        psi += perturbed_weights[left, right].sum()
    return psi / M


def get_distribution(log_weights, samples):
    N, _ = log_weights.shape
    P = np.zeros(N+1)

    j = len(samples)
    I = set(range(N))
    sample_from = I - set(samples)
    prev_bound = get_upper_bound(log_weights[j:, list(sample_from)])

    for v in sample_from:
        s = list(sample_from - {v})
        w = log_weights[j, v]
        u = get_upper_bound(log_weights[j+1:, s])
        P[v] = np.exp(u + w - prev_bound)

    P[-1] = 1 - P.sum()
    return P

def sampler(bmatrix):
    r = count = neg = 0
    N, _ = bmatrix.shape
    log_weights = np.log(bmatrix)
    reject = N
    samples = []
    while len(samples) < N - 1:
        count += 1
        P = get_distribution(log_weights, samples)
        if P[-1] < 0:
            neg += 1
            P[-1] = 0
            P=P/P.sum()

        sampleId = npr.choice(range(len(P)), p=P)
        if sampleId == reject:
            r += 1
            samples = []
        else:
            samples.append(sampleId)

    print("\n\nR=%d\tneg=%.2f%%" % (r,100*neg/count))
    return samples

def main():
    n = 7
    N = 10
    for n in range(5,30):
        T = np.zeros(N)
        for i in range(N):
            weights = npr.randint(1,10, size=(n,n))
            start = time()
            samples = sampler(weights)
            end = time()
            T[i] = end-start
        print("Average running time for n=%d is t=%.2f s" % (n, T.mean()))

    print('samples', samples)

if __name__ == '__main__':
    main()