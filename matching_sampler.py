from time import time

import numpy as np
import numpy.random as npr
from ryser import rysers
from scipy.optimize import linear_sum_assignment

EULER = 0.5772156649

def GumbelNoise(N):
    return -np.log(-np.log(npr.uniform(size=N))) - EULER

def get_bounds(log_weights):
    M = 10
    U = L = 0
    n = log_weights.size

    for i in range(M):
        noise = GumbelNoise(log_weights.shape)
        UPW = log_weights + noise
        left, right = linear_sum_assignment(UPW, maximize=True)
        U += UPW[left, right].sum()/ M

        LPW = log_weights + noise/n
        left, right = linear_sum_assignment(LPW, maximize=True)
        L += LPW[left, right].sum()/ M
    return U, L


def get_distribution(log_weights, samples):
    N, _ = log_weights.shape
    P = np.zeros(N+1)

    j = len(samples)
    I = set(range(N))
    sample_from = I - set(samples)
    U, L = get_bounds(log_weights[j:, list(sample_from)])

    for v in sample_from:
        s = list(sample_from - {v})
        w = log_weights[j, v]
        u, _ = get_bounds(log_weights[j+1:, s])
        # TODO: u + w (- U?)
        P[v] = np.exp(u + w - U)

    P[-1] = 1 - P.sum()
    return P, U, L

def sampler(bmatrix):
    r = count = neg = 0
    N, _ = bmatrix.shape
    log_weights = np.log(bmatrix)
    reject = N
    samples = []
    u = 0
    while len(samples) < N - 1:
        count += 1
        P, U, L = get_distribution(log_weights, samples)
        u += U
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

    return samples, 1-r/count, neg/count, u/count

def main():
    # n = 7
    # N = 10
    # for n in range(5,30):
    #     T = np.zeros(N)
    #     for i in range(N):
    #         weights = npr.randint(1,10, size=(n,n))
    #         start = time()
    #         samples = sampler(weights)
    #         end = time()
    #         T[i] = end-start
    #     print("Average running time for n=%d is t=%.2f s" % (n, T.mean()))

    M = npr.uniform(size=(20,20))

    t0 = time()
    Z = rysers(M)
    t1 = time()
    rysers_runtime = (t1-t0)

    t0 = time()
    samples, accept, negative, upper_bound = sampler(M)
    t1 = time()
    sampler_runtime = (t1-t0)

    c = 45
    print()
    print("="*c)
    print("| method \t | runtime \t | log Z    |")
    print("-"*c)
    print("| %s \t | %.3f s \t | %.5f |" % ('rysers', rysers_runtime, np.log(Z)))
    print("-"*c)
    print("| %s \t | %.3f s \t | %.5f |" % ('sampler', sampler_runtime, upper_bound))
    print("-"*c)


if __name__ == '__main__':
    main()
