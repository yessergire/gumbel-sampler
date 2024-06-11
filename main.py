
import numpy as np
from plots import (full_order_Gumbel_PM_size_wrt_error,
                   full_order_Gumbel_PM_size_wrt_time,
                   low_order_Gumbel_PM_size_wrt_error,
                   low_order_Gumbel_PM_size_wrt_time)

if __name__ == '__main__':
    with np.errstate(divide="ignore"):
        N = 10
        # full_order_Gumbel_PM_size_wrt_time(N)
        # full_order_Gumbel_PM_size_wrt_error(7, M=20)

        # low_order_Gumbel_PM_size_wrt_time(N)
        low_order_Gumbel_PM_size_wrt_error(N, M=50)