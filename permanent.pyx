from scipy.optimize import linear_sum_assignment
import numpy as np
import numpy.random as npr


def gamma(k):
    if k == 0:
        return 0
    return np.exp(np.log(np.arange(1, k+1)).sum()/k)


def soules(double[:,:] M):
    n = M.shape[0]
    cdef double[:] Delta = np.array([gamma(j+1) - gamma(j) for j in range(n)])

    approx = 0
    for i in range(n):
        a = sorted(M[i,:], reverse=True)
        approx += np.log(np.sum([ (a[j] * Delta[j]) for j in range(n) ]))
    return np.exp(approx)


def rysers(double[:,:] M):
    n = M.shape[0]
    nc = M.shape[1]
    if not n is nc:
        print("M must be square")

    cdef double[:] row_sums = np.zeros(n)
    cdef double subset_sum = 0
    cdef long s = 1

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
