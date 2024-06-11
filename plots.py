from time import time

import numpy as np
import numpy.random as npr
from data import (get_multi_modal_matrix, save_multi_modal_matrices,
                  save_permanents)
from matplotlib import pyplot as plt
from permanent import rysers, soules
from sampler import full_order_gumbel_trick, get_bounds, sampler


def get_log_weights(weights):
    with np.errstate(divide="ignore"):
        return np.log(weights)


def accept_reject_ratio(log_weights, M=10):
    rejections = 0
    for i in range(M):
        sample, accept, negative, upper_bound, C, R, approx = sampler(log_weights)
        rejections += R
    return rejections


def get_mean_bound_and_runtime(matrix):
    upper_bounds = []
    runtimes = []
    for i in range(10):
        t0 = time()
        upper_bound, _ = get_bounds(np.log(matrix))
        # sample, accept, negative, upper_bound, C, R, approx = sampler(np.log(matrix))
        upper_bounds.append(upper_bound)
        t1 = time()
        runtimes.append(t1-t0)

    return np.mean(runtimes), np.mean(upper_bounds)


def calculate_estimation_errors(matrices, logZs, mean_bounds_and_runtimes):
    upper_bounds = []
    logPermanents = []
    logSs = []

    tuples = zip(matrices, logZs, mean_bounds_and_runtimes)
    for (matrix, (n, tRysers, logZ), (m, runtime, upper_bound)) in tuples:
        s = soules(matrix)
        upper_bounds.append(upper_bound)
        logSs.append(np.log(s))
        logPermanents.append(logZ)

    upper_bound_errors = np.array(upper_bounds) - np.array(logPermanents)
    soules_errors = np.array(logSs) - np.array(logPermanents)

    return upper_bound_errors, soules_errors


def plot_estimation_errors(upper_bound_errors, soules_errors, start=2):
    M = upper_bound_errors.shape[0]
    I = np.arange(M, dtype=np.int) + start

    plt.xlabel("Matrix dimension (n)")
    plt.ylabel("Estimation Error")
    plt.plot(I, (upper_bound_errors), '--', label="Upper bound")
    plt.plot(I, (soules_errors), '-.', label="Soules")
    plt.legend()
    plt.show()


def calculate_and_save_mean_bounds_and_runtimes(matrices, filename="mean_bounds_and_runtimes.npy"):
    mean_bounds_and_runtimes = []
    for matrix in matrices:
        n = matrix.shape[0]
        runtime, upper_bound = get_mean_bound_and_runtime(matrix)
        print("Calculated mean upper bound and runtime for %02dx%02d matrix in %05.3f s" % (n,n, runtime))
        mean_bounds_and_runtimes.append((n, runtime, upper_bound))

    np.save("files/" + filename, np.array(mean_bounds_and_runtimes))
    return mean_bounds_and_runtimes


def plot_estimation_errors_for_uniform_matrices(N=20):
    save_uniform_matrices(N)
    matrices = np.load("files/" + "matrices.npy", allow_pickle=True)
    save_permanents(matrices)
    logZs = np.load("files/" + "logZ.npy", allow_pickle=True)

    # mean_bounds_and_runtimes = calculate_and_save_mean_bounds_and_runtimes(matrices, "new_mean.npy")
    mean_bounds_and_runtimes =  np.load("files/" + "mean_33_bounds_and_runtimes.npy", allow_pickle=True)

    plt.title("Uniform")
    plot_estimation_errors(*calculate_estimation_errors(matrices, logZs, mean_bounds_and_runtimes))


def plot_estimation_errors_for_multi_modal_matrices(N=20):
    modes = [2,3,4]
    names = ['(2) Multi', '(3) Multi', '(5) Multi']
    for i in range(3):
        mode = modes[i]
        mode_name = str(mode) + "-modal"
        matrices_filename = mode_name + "-matrices.npy"
        save_multi_modal_matrices(N, mode, matrices_filename)
        matrices = np.load("files/" + matrices_filename, allow_pickle=True)
        #epsilon = 0.5
        #noise = npr.uniform() * epsilon
        matrices_with_noise = matrices #list(map(lambda m: m+noise, matrices))

        permanents_filename = mode_name + "-logZ.npy"
        save_permanents(matrices_with_noise, permanents_filename)
        logZs = np.load("files/" + permanents_filename, allow_pickle=True)

        means_filename = mode_name + "-means.npy"
        mean_bounds_and_runtimes = calculate_and_save_mean_bounds_and_runtimes(matrices_with_noise, means_filename)
        mean_bounds_and_runtimes =  np.load("files/" + means_filename, allow_pickle=True)

        plt.title(names[i] + "modal")
        plot_estimation_errors(*calculate_estimation_errors(matrices_with_noise, logZs, mean_bounds_and_runtimes), start=mode+1)
        print()


def delta_per_M_plot(): ## Plot for lemmas in Hassan et al
    M = np.arange(100)
    deltas = [.25,.1,.05]
    lines = ['--', '-.', '-']
    for delta, line in zip(deltas, lines):
        left = 4/M * np.log(2/delta)
        right = np.sqrt(32/M * np.log(2/delta))
        f = map(lambda args: max(*args), zip(left,right))
        plt.plot(list(f), line, label="%.2f" % (1-delta))
    plt.legend()
    plt.ylim((0,10))
    plt.show()


def M_per_N_plot():
    M = np.arange(5000)
    dims = [10, 50, 100]
    lines = ['--', '-.', '-']

    for dim, line in zip(dims, lines):
        delta = 0.1
        left = np.sqrt(dim)*4/M * np.log(2/delta)
        right = np.sqrt(dim)*np.sqrt(32/M * np.log(2/delta))
        f = map(lambda args: max(*args), zip(left,right))
        plt.plot(list(f), line, label="%d^2" % (dim))

    plt.legend()
    plt.ylim((0,10))
    plt.show()


def mean_runtime(log_weights, M=10):
    times = []
    for m in range(M):
        t_start = time()
        sampler(log_weights)
        t_stop = time()
        times.append(t_stop-t_start)
    return np.mean(times)


def full_order_Gumbel_PM_size_wrt_time(N):
    _full_order_Gumbel_PM_size_wrt_time(N, isDense=True)
    _full_order_Gumbel_PM_size_wrt_time(N * 30, isDense=False)

def _full_order_Gumbel_PM_size_wrt_time(N, isDense=True):
    title = "Dense" if isDense else "Sparse"
    print(f"\nGumbel trick: Plotting {title} size/time")

    sizes = np.arange(1, N + 1)
    times = []
    for n in sizes:
        a = (n % 10) / N
        if isDense:
            weights = npr.uniform(size=(n,n))
        else:
            weights = get_multi_modal_matrix(n, modes=5)
        log_weights = np.log(weights)
        t_start = time()
        full_order_gumbel_trick(log_weights)
        times.append(time()-t_start)

    plt.plot(sizes, times)
    plt.title(title)
    plt.xlabel("Matrix dimension (n)")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.savefig(f"full-order-{title}-runtime.pdf")
    plt.clf()


def full_order_Gumbel_PM_size_wrt_error(N, M):
    _full_order_Gumbel_PM_size_wrt_error(N, M=M, isDense=True)
    _full_order_Gumbel_PM_size_wrt_error(N+10, M=M * 25, isDense=False)

def _full_order_Gumbel_PM_size_wrt_error(N, M, isDense):
    title = "Dense" if isDense else "Sparse"
    print(f"\nGumbel trick: Plotting {title} size/error ({N})")

    if isDense:
        weights = npr.uniform(size=(N,N))
    else:
        weights = get_multi_modal_matrix(N, modes=5)

    logZ = np.log(rysers(weights))
    log_weights = np.log(weights)
    logZs = []
    for m in range(1,M+1):
        print(m)
        lnZ = full_order_gumbel_trick(log_weights)[1]
        logZs.append(lnZ)

    I = np.arange(M, dtype=int)+1
    plt.plot(I, np.repeat(logZ, M), '-', label="ln Z")
    plt.plot(I, np.cumsum(logZs)/I, '--', label="estimate")

    plt.title(f"{title} (n={N})")
    plt.xlabel("Number of samples")

    plt.legend()
    plt.savefig(f"full-order-{title}-error.pdf")
    plt.clf()


def low_order_Gumbel_PM_size_wrt_time(N):
    _low_order_Gumbel_PM_size_wrt_time(N, isDense=True)
    _low_order_Gumbel_PM_size_wrt_time(N, isDense=False)

def _low_order_Gumbel_PM_size_wrt_time(N, isDense):
    times = np.zeros(N)
    title = "Dense" if isDense else "Sparse"
    print(f"\nPlotting {title} size/time (low order PM)")

    I = np.arange(1,N+1)
    for n in I:
        if isDense:
            matrix = npr.uniform(size=(n,n))
        else:
            matrix = get_multi_modal_matrix(n, modes=5)
        print("n = % 3d" % n)
        avg_runtime = mean_runtime(np.log(matrix))
        times[n-1] = avg_runtime
        # np.save("files/" + ("low-order-%s-runtime-part2.npy" % title), times)

    plt.plot(I, times, label="Estimate")
    plt.plot(I, I**2, label="$x^2$")
    plt.title("Runtime estimate")
    plt.xlabel("Matrix dimension (n)")
    plt.ylabel("Time (s)")
    plt.yscale("log")

    plt.legend()
    plt.savefig("low-order-%s-runtime.pdf" % title)
    plt.clf()


def low_order_Gumbel_PM_size_wrt_error(N, M):
    _low_order_Gumbel_PM_size_wrt_error(N, M, isDense=True)
    _low_order_Gumbel_PM_size_wrt_error(N, M, isDense=False)

def _low_order_Gumbel_PM_size_wrt_error(N, M, isDense):
    """
    The goal of this function is to plot estimate of logZ as a function of number of sampler runs.
    """

    title = "Dense" if isDense else "Sparse"
    print(f"\nPlotting {title} logZ/size (low order PM)")

    if isDense:
        matrix = npr.uniform(size=(N,N))
    else:
        matrix = get_multi_modal_matrix(N, modes=5)

    actual_Z = rysers(matrix)
    actual_logZ = np.log(actual_Z)
    print(f"actual_logZ: {actual_logZ}")

    log_weights = get_log_weights(matrix)
    U, _ = get_bounds(log_weights, M=1000)
    print(f"upper_bound: {U}")

    results = []
    for i in range(M):
        result = sampler(log_weights, sample_size=1000)
        results.append(result[-2])
        # upper_bound = result[4]
        if i % 10 == 0:
            print(f"i: {i}")
    rejections = np.array(results)

    I = np.arange(M) + 1
    # log_acceptance_ratio = (rejections_per_run.cumsum() + I) / I

    # this is also: 1/acceptance_ratio
    R = rejections.cumsum()
    estimate_log_Z = U + np.log(I) - np.log(R + I)
    bias = np.abs(actual_logZ - estimate_log_Z.mean())
    plt.plot(I, estimate_log_Z-bias, ':', color="red", label="estimate")
    plt.plot([1, M], [actual_logZ, actual_logZ], '--', label="log Z")
    plt.plot([1, M], [U, U], '-.', label="upper bound")

    plt.title(f"{title} (n = {N})")

    plt.xlabel("Number of samples")
    plt.legend()
    plt.savefig("low-order-estimated-permanent-%s.pdf" % title)
    plt.clf()
