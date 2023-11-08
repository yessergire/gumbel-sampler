from time import time

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from permanent import rysers, soules

from data import save_permanents
from sampler import EULER, full_perturb_MAP, get_bounds, inv_g, sampler


def accept_reject_ratio(log_weights, M=10):
    rejections = 0
    for i in range(M):
        sample, accept, negative, upper_bound, C, R, approxes = sampler(log_weights)
        rejections += R
    return rejections


def get_mean_bound_and_runtime(matrix):
    upper_bounds = []
    runtimes = []
    for i in range(10):
        t0 = time()
        upper_bound, _ = get_bounds(np.log(matrix))
        # sample, accept, negative, upper_bound, C, R, approxes = sampler(np.log(matrix))
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
    # save_uniform_matrices(N)
    matrices = np.load("files/" + "matrices.npy", allow_pickle=True)
    # save_permanents(matrices)
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
        #save_multi_modal_matrices(N, mode, matrices_filename)
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


def main2():
    N = 12
    # matrix = get_multi_modal_matrix(N, 3) # + npr.uniform(size=(N,N)) * 0.10

    matrix = npr.uniform(size=(N,N))
    Z = rysers(matrix)
    logZ = np.log(Z)

    S = soules(matrix)
    logS = np.log(S)

    sample, accept, negative, upper_bound, C, R, approxes = sampler(np.log(matrix))
    E_psi_0 = logZ - upper_bound
    print("N = %d; \t C = %d \t N/C = %.2f" % (negative, C, negative/C))
    print("negative",negative)

    print("Z = %.5f \t E(U) = %.5f \t P(A) = %.5f \t C = %d \t R = %d  \t E(S) = %.5f" % (Z, E_psi_0, np.exp(E_psi_0), C, R, logZ - logS))

    c = len(approxes)
    plt.plot(np.arange(c), np.repeat(logZ, c), '-', label="log Z")
    plt.plot(np.arange(c), np.repeat(upper_bound, c), '-.', label="upper bound")
    plt.plot(np.arange(c), approxes, '--', label="approximation")
    plt.plot(np.arange(c), np.repeat(logS, c), '-', label="Soules")
    plt.title("Uniform (n = %d)" % N)
    plt.xlabel("Iterations")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def delta_per_M_plot(): ## Plot for lemmas in Hassan et al
    M = np.arange(1000)
    deltas = [.25,.1,.05]
    lines = ['--', '-.', '-']
    for delta, line in zip(deltas, lines):
        left = 4/M * np.log(2/delta)
        right = np.sqrt(32/M * np.log(2/delta))
        f = map(lambda args: max(*args), zip(left,right))
        plt.plot(list(f), line, label="%.2f" % (1-delta))
    plt.legend()
    plt.ylim((0,20))
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


def full_order_Gumbel_PM_size_wrt_time(title, dims, generate_matrix):
    print("\n" + "Plotting input size wrt time (full order PM)")
    times = []
    for n in dims:
        print("n = % 3d" % n)
        matrix = generate_matrix(n)
        log_weights = np.log(matrix)
        t_start = time()
        full_perturb_MAP(log_weights)
        t_stop = time()
        times.append(t_stop-t_start)

    plt.plot(dims, times)
    plt.title(title)
    plt.xlabel("Matrix dimension (n)")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.savefig("full-order-%s-runtime.pdf" % title)
    plt.clf()


def full_order_Gumbel_PM_size_wrt_error(title, matrix, M=100):
    n = matrix.shape[0]
    logZ = np.log(rysers(matrix))
    log_weights = np.log(matrix)
    samples = []
    # samples_mapped = []
    results = []
    # results_mapped = []
    sizes = []
    for m in range(1,M+1):
        sizes.append(m)
        logZs = [full_perturb_MAP(log_weights)[1] for i in range(M)]
        samples.append(logZs)
        # inv_Zs = list(map(inv_g, logZs))
        # logZ --inv_g-> Z
        # Zs = list(map(lambda x: 1/x, inv_Zs))
        # Z --inv-> Z^(-1) --g-> logZ
        # lnZs = np.log(Zs)-EULER
        # samples_mapped.append(lnZs+.001)
        results.append(np.mean(samples))
        # results_mapped.append(np.mean(samples_mapped))

    plt.plot(sizes, np.repeat(logZ, len(sizes)), '-', label="ln Z")
    plt.plot(sizes, results, '--', label="estimate")
    # plt.plot(sizes, results_mapped, '--', label="estimate mapped")
    plt.title("%s (n=%d)" % (title, n))
    plt.xlabel("Number of samples")
    plt.legend()
    plt.savefig("full-order-%s-error.pdf" % title)
    plt.clf()


def mean_runtime(log_weights, M=10):
    times = []
    for m in range(M):
        t_start = time()
        sampler(log_weights)
        t_stop = time()
        times.append(t_stop-t_start)
    return np.mean(times)


def low_order_Gumbel_PM_size_wrt_time(title, dims, generate_matrix):
    print("\n" + "Plotting input size wrt time (low order PM)")
    # times = np.zeros(dims[-1])
    old_times = np.load("files/" + ("low-order-%s-runtime-part2.npy" % title), allow_pickle=True)
    # times[:old_times.size] = old_times
    # new_dims = dims[old_times.size:]
    # for n in new_dims:
    #     print("n = % 3d" % n)
    #     matrix = generate_matrix(n)
    #     avg_runtime = mean_runtime(np.log(matrix))
    #     times[n-1] = avg_runtime
    #     np.save("files/" + ("low-order-%s-runtime-part2.npy" % title), times)

    plt.plot(dims, old_times, label="Estimate")
    plt.plot(dims, dims**2, label="$x^2$")
    plt.title("Runtime estimate")
    plt.xlabel("Matrix dimension (n)")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.savefig("low-order-%s-runtime.pdf" % title)
    plt.legend()
    plt.show()
    plt.clf()


def low_order_Gumbel_PM_size_wrt_error(title, n, matrix, M=50):
    Z = rysers(matrix)
    logZ = np.log(Z)
    log_weights = np.log(matrix)
    upper_bound, _ = get_bounds(log_weights, 100)

    rejections = []
    for i in range(M):
        sample, accept, negative, _, C, r, _ = sampler(log_weights, upper_bound, 100)
        rejections.append(r)
    rejections = np.array(rejections)

    indices = np.arange(M) + 1
    pA = (rejections.cumsum()+indices) / indices

    average_rejections = (rejections.cumsum()) / indices
    plt.plot(indices, upper_bound - np.log(average_rejections + 1), '--', label="estimate")
    plt.plot(indices, np.repeat(logZ, M), '-', label="log Z")
    plt.plot(indices, np.repeat(upper_bound, M), '-.', label="upper bound")

    plt.title("%s (n = %d)" % (title, n))

    plt.xlabel("Number of samples")
    plt.legend()
    plt.savefig("low-order-%s-error.pdf" % title)
    plt.clf()


def plot_estimation_error(matrix):
    """
    The idea is to plot upper bound as a function of number of samples (M)
    X-axis: we've got X and on the Y axis we've got E[U(matrix)]
    """
    Z = rysers(matrix)
    logZ = np.log(Z)
    log_weights = np.log(matrix)
    M = 2000
    U, upper_bounds = get_bounds(log_weights, M)

    indices = np.arange(M)+1
    plt.plot(indices, np.repeat(U/logZ, M), label="U")
    # plt.plot(indices, np.repeat(logZ, M), '--', label="log Z")
    plt.plot(indices, (upper_bounds / (indices*logZ)), label="upper bound")

    plt.title("Upper bound")
    plt.xlabel("Number of samples")
    plt.legend()
    plt.savefig("low-order-upper-bound.pdf")
    plt.clf()
