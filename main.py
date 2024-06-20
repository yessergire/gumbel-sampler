
import numpy as np
import numpy.random as npr
from plots import (full_order_Gumbel_PM_size_wrt_error,
                   full_order_Gumbel_PM_size_wrt_time,
                   low_order_Gumbel_PM_size_wrt_error,
                   low_order_Gumbel_PM_size_wrt_time, plot_estimation_errors)

if __name__ == '__main__':
    npr.seed(100)
    with np.errstate(divide="ignore"):
        N = 10
        M = 100
        plot_estimation_errors(N, M=200)
        full_order_Gumbel_PM_size_wrt_time(N)
        full_order_Gumbel_PM_size_wrt_error(8, M)

        low_order_Gumbel_PM_size_wrt_time(N)
        low_order_Gumbel_PM_size_wrt_error(N, M)

