import numpy as np
import numpy.random as npr
from scipy.optimize import linear_sum_assignment

EULER = 0.5772156649


def inv_g(x):
    return np.exp(-x-EULER)


def gumbel(x):
    return -np.ln(x)-EULER


def exponential(x, alpha=1):
    # weibull (alpha > 0) or frechet (-1 < alpha < 0)
    return x**alpha


def pareto(x):
    return np.exp(x)


def tail(x, t):
    return x > t


def GumbelNoise(N=1):
    return -np.log(-np.log(npr.uniform(size=N))) - EULER


def sampleWithGumbel(log_weights):
    n = log_weights.size
    perturbed_weights = log_weights + GumbelNoise(n)
    max_pair = max(zip(np.arange(n), perturbed_weights), lambda x: x[1])
    return max_pair


def get_bounds(log_weights, M=50):
    U = 0
    n = log_weights.shape[0]

    upper_bounds = []
    for i in range(M):
        noise = GumbelNoise((n,n)) # noise for each edge (=2dim)
        UPW = log_weights + noise
        try:
            left, right = linear_sum_assignment(UPW, maximize=True)
        except:
            return -np.inf, np.inf
        U += UPW[left, right].sum()
        upper_bounds.append(U)

        # LPW = log_weights + noise/n
        # left, right = linear_sum_assignment(LPW, maximize=True)
        # L += LPW[left, right].sum()
    return U / M, upper_bounds


# TODO: make sure this is the 2D case
def get_distribution(log_weights, samples, M, U=None):
    N = log_weights.shape[0]
    P = np.zeros(N+1)

    j = len(samples)
    I = set(range(N))
    sample_from = I - set(samples)
    if U == None:
        U, _ = get_bounds(log_weights[j:, list(sample_from)], M)
    # W = sum([log_weights[i, samples[i]] for i in range(j)])
    # U, L = get_bounds(log_weights)

    for v in sample_from:
        s = list(sample_from - {v})
        w = log_weights[j, v]
        u, _ = get_bounds(log_weights[j+1:, s])
        # TODO: u + w (- U?)
        P[v] = np.exp(u + w - U)

    P[-1] = 1 - sum(P)
    return P, U


def sampler(log_weights, upper_bound=None, M=10):
    rejections = iterations = neg = 0
    N = log_weights.shape[0]
    reject_symbol = N
    samples = []
    approxes = []

    if upper_bound == None:
        upper_bound, _ = get_bounds(log_weights)
    upper_bound, _ = get_bounds(log_weights)

    c = 0
    while len(samples) < N:
        c += 1
        if len(samples) == 0:
            iterations += 1

        if len(samples) == 0:
            U = upper_bound
        else:
            U = None

        P, _ = get_distribution(log_weights, samples, M, U)
        # uSumE += upper

        if P[-1] < 0: # TODO: How can this be < 0
            neg += 1
            P[-1] = 0
            P=P/np.sum(P)

        sampleId = npr.choice(range(len(P)), p=P)

        if sampleId == reject_symbol:
            rejections += 1
            samples = []
        else:
            samples.append(sampleId)

        pA = (iterations - rejections) / iterations
        approxes.append(upper_bound + np.log(pA))

    return samples, 1-rejections/iterations, neg, upper_bound, c, rejections, np.array(approxes)


def full_perturb_MAP(log_weights, samples=[]):
    n = log_weights.shape[0]
    i = len(samples)
    results = []
    if i < n:
        for j in range(n):
            if j in samples: continue
            if np.isfinite(log_weights[i,j]):
                result_j = full_perturb_MAP(log_weights, samples + [j])
                if len(result_j) > 0:
                    results.append(result_j)
            if len(results) > 0:
                max_result = max(results, key=lambda parts: parts[1])
            else:
                max_result = []
        return max_result
    weight_sum = sum([log_weights[i, samples[i]] for i in range(n)])
    result = (samples, weight_sum + GumbelNoise())
    return result
