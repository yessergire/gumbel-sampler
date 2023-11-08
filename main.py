
import numpy as np
import numpy.random as npr

from data import get_multi_modal_matrix
from plots import (full_order_Gumbel_PM_size_wrt_error,
                   full_order_Gumbel_PM_size_wrt_time, plot_estimation_error)

if __name__ == '__main__':
    n = 7

    # full_order_Gumbel_PM_size_wrt_time("Dense", np.arange(1,11), lambda n: npr.uniform(size=(n,n)))
    full_order_Gumbel_PM_size_wrt_time("Sparse", np.arange(1,11)*30, lambda n: get_multi_modal_matrix(n, modes=10))

    # full_order_Gumbel_PM_size_wrt_error("Dense", npr.uniform(size=(n,n)), 200)
    full_order_Gumbel_PM_size_wrt_error("Sparse", get_multi_modal_matrix(n, modes=10), 200)

    # J = np.arange(20) + 1
    # low_order_Gumbel_PM_size_wrt_time("Dense", J, lambda n: npr.uniform(size=(n,n)))

    # matrix = npr.uniform(size=(20,20))
    # plot_estimation_error(matrix)
    # low_order_Gumbel_PM_size_wrt_time("Sparse", J, lambda n: get_multi_modal_matrix(n, modes=5))

    # low_order_Gumbel_PM_size_wrt_error("Dense", n, npr.uniform(size=(n,n)), M=100)
    # low_order_Gumbel_PM_size_wrt_error("Sparse", n, get_multi_modal_matrix(n, modes=10), M=200)
 
    # TODO:
    # 1) Save results in npy format
    # 2) then plot the results (with right limits & scales)
    # 3) Save to a pdf file
