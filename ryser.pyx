import numpy as np


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
