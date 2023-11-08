from time import time

import numpy as np
import numpy.random as npr
from permanent import rysers


def get_multi_modal_matrix(n, modes):
    matrix = np.zeros((n,n))
    I = np.arange(n)
    matrix[I,I] = 1
    for i in range(0, min(n-1, modes-1)):
        matrix[i, i+1] = matrix[i+1, i] = 1
    return matrix


def save_multi_modal_matrices(N, modes, filename="matrices.npy"):
    matrices = []
    for n in range(max(modes+1, 2), N+1):
        matrix = get_multi_modal_matrix(n, modes)
        matrices.append(matrix)
    np.save("files/" + filename, matrices)


def save_uniform_matrices(N, filename="matrices.npy"):
    matrices = []
    for n in range(2, N+1):
        matrices.append(npr.uniform(size=(n,n)))
    np.save("files/" + filename, matrices)


def save_permanents(matrices, filename="logZ.npy"):
    permanents = []
    for matrix in matrices:
        n = matrix.shape[0]
        t0 = time()
        logZ = np.log(rysers(matrix))
        t1 = time()
        permanents.append([n, t1-t0, logZ])
        print("Calculated log permanent of a %dx%d matrix in %.4f s" % (n, n, t1-t0),)
    np.save("files/" + filename, np.array(permanents))

