from time import time

import numpy as np
import numpy.random as npr
from sampler import rysers, sampler


def print_rysers_runtime(N):
    matrix = npr.uniform(size=(N,N))

    t0 = time()
    Z = rysers(matrix)
    t1 = time()

    c = 24
    print()
    print("="*c)
    print("|     Ryser's method   |")
    print("-"*c)
    print("|    runtime |  log Z  |")
    print("-"*c)
    print("| % 7.4f s  | % 7.2f |" % ((t1-t0), np.log(Z)))
    print("-"*c)

def get_mean_bound_and_runtime(N):
    matrix = npr.uniform(size=(N,N))

    upper_bounds = []
    runtimes = []
    for i in range(10):
        t0 = time()
        samples, accept, negative, upper_bound = sampler(matrix)
        upper_bounds.append(upper_bound)
        t1 = time()
        runtimes.append(t1-t0)

    return np.mean(runtimes), np.mean(upper_bounds)


def main():

    c = 30
    print()
    print("+" + "-"*c + "+")
    print("| % 20s % 7s |" % ("Gibbs method", ""))
    print("+" + "-"*c + "+")
    print("|   N  |    runtime |   log Z  |")
    print("+" + "-"*c + "+")

    # for n in range(2,31):
    n = 29
    runtime, upper_bound = get_mean_bound_and_runtime(n)

    print("| % 3s  | % 7.3f s  | % 7.2f  |" % (n, runtime, upper_bound))
    print("+" + "-"*c + "+")


if __name__ == '__main__':
    main()
