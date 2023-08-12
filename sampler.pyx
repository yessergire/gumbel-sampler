from scipy.optimize import linear_sum_assignment
import numpy as np
import numpy.random as npr

EULER = 0.5772156649

def GumbelNoise(N):
    return -np.log(-np.log(npr.uniform(size=N))) - EULER


def get_bounds(double[:,:] log_weights):
    M = 10
    U = L = 0
    cdef int n = log_weights.shape[0]
    cdef int m = log_weights.shape[1]
    cdef long[:] left, right

    for i in range(M):
        noise = GumbelNoise((n, m))
        UPW = log_weights + noise
        left, right = linear_sum_assignment(UPW, maximize=True)
        U += UPW[left, right].sum() / M

        #LPW = log_weights + noise/n
        #left, right = linear_sum_assignment(LPW, maximize=True)
        #L += LPW[left, right].sum() / M
    return U, L


def get_distribution(log_weights, samples):
    N = log_weights.shape[0]
    cdef double[:] P = np.zeros(N+1)

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
        sum = P[v]

    P[-1] = 1 - np.sum(P)
    return P, U, L


def sampler(double[:,:] M):
    r = count = neg = 0
    N = M.shape[0]
    log_weights = np.log(M)
    reject_symbol = N
    samples = []
    u = 0
    while len(samples) < N - 1:
        count += 1
        P, U, L = get_distribution(log_weights, samples)
        u += U
        if P[-1] < 0:
            neg += 1
            P[-1] = 0
            P=P/np.sum(P)

        sampleId = npr.choice(range(len(P)), p=P)
        if sampleId == reject_symbol:
            r += 1
            samples = []
        else:
            samples.append(sampleId)

    return samples, 1-r/count, neg/count, u/count


def rysers(double[:,:] M):
    n = M.shape[0]
    nc = M.shape[1]
    if not n is nc:
        print("M must be square")

    cdef double[:] row_sums = np.zeros(n)
    subset_sum = 0
    s = 1

    even = lambda x: x % 2 == 0
    sign = lambda x: 1 if even(x) else -1

    while s < 2 ** n:
        mask = s ^ (s // 2) ^ (s - 1) ^ ((s - 1) // 2)
        j = 0
        while ((2 ** j) & mask) == 0: j += 1

        if ((s ^ (s//2)) & mask) != 0:
            for i in range(n):
                row_sums[i] += M[i][j]
        else:
            for i in range(n):
                row_sums[i] -= M[i][j]
        prod = 1
        c = 0
        for i in range(n):
            prod *= row_sums[i]
            c += ((s ^ (s // 2)) & (2 ** i)) > 0
        subset_sum += sign(c) * prod
        s += 1
    return sign(n) * subset_sum
