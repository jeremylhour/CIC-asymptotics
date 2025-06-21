"""
Functions related to estimation of empirical CDF,
mainly smoothed empirical CDF as in Shorack and Wellner.

Created on Mon Nov  9 13:43:00 2020

@author : jeremylhour
"""

import numpy as np
from numba import njit


# ------------------------------------------------------------------------------------
# EMPIRICAL CDF
# ------------------------------------------------------------------------------------
@njit
def ranks_and_antiranks(x):
    """
    ranks_and_antiranks:
        returns ranks and antiranks for an array of points

    Args:
        points (np.array): vector of points of dimension (n,).
    """
    antiranks = np.argsort(x)
    ranks = np.argsort(antiranks)
    return ranks, antiranks


@njit
def smoothed_empirical_cdf(x, data):
    """
    smoothed_empirical_cdf:
        Smoothed empirical CDF as in Shorack and Wellner (p. 86), but extended to non-bounded support.
        Linear extension outside the support using the nearest linear parts.

    Args:
        x (np.array): new points for which the value is returned.
        data (np.array): points used to compute the smoothed empirical CDF.
    """
    n = len(data)
    sorted_data = np.sort(data)

    ### Compute extreme values outside the support (cf. e-mail Xavier 09/11),
    ### by extending the affine smoothing to the origin or to 1.
    unique_vals = np.unique(sorted_data)

    ### LOWER BOUND ###
    u0 = unique_vals[0]
    u1 = unique_vals[1]

    # Count elements (use searchsorted to avoid full scans)
    y0 = np.searchsorted(sorted_data, u0, side="right") - np.searchsorted(
        sorted_data, u0, side="left"
    )
    y1 = np.searchsorted(sorted_data, u1, side="right")

    b1 = (y1 - y0) / ((n + 1) * (u1 - u0))
    a1 = y0 / (n + 1) - b1 * u0
    lb = -a1 / b1

    ### UPPER BOUND ###
    u_last = unique_vals[-1]
    u_second_last = unique_vals[-2]

    # a. accounting for duplicates
    y_last = np.searchsorted(sorted_data, u_last, side="right")
    y_second_last = np.searchsorted(sorted_data, u_second_last, side="right")

    bn = (y_last - y_second_last) / ((n + 1) * (u_last - u_second_last))
    an = y_last / (n + 1) - bn * u_last
    ub = (1 - an) / bn

    # new array with upper and lower bounds
    extended_data = np.concatenate((np.array([lb]), sorted_data, np.array([ub])))

    result = np.empty(len(x))
    for i in range(len(x)):
        xi = x[i]
        if xi < lb:
            result[i] = 0.0
        elif xi > ub:
            result[i] = 1.0
        else:
            idx = np.searchsorted(extended_data, xi) - 1
            x0 = extended_data[idx]
            x1 = extended_data[idx + 1]
            b = 1.0 / ((n + 1) * (x1 - x0))
            a = idx / (n + 1) - b * x0
            result[i] = a + b * xi
    return result
