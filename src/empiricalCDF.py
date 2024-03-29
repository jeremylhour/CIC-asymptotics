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
    ranks = np.array([np.sort(x).index(i) for i in x])
    return ranks, np.argsort(ranks)

@njit
def smoothed_ecdf(new_points, data):
    """
    smoothed_ecdf:
        Smoothed empirical CDF as in Shorack and Wellner (p. 86), but extended to non-bounded support.
        Linear extension outside the support using the nearest linear parts.
    
    Args:
        new_points (np.array): new points for which the value is returned.
        data (np.array): points used to compute the smoothed empirical CDF.
    """
    n_points = len(data)
    sorted_x = np.sort(data)
    
    ### Compute extreme values outside the support (cf. e-mail Xavier 09/11),
    ### by extending the affine smoothing to the origin or to 1.
    unique_sorted_x = np.unique(sorted_x)
    
    ### LOWER BOUND ###

    # a. accounting for duplicates
    y_0 = np.sum(data == unique_sorted_x[0])
    y_1 = np.sum(data <= unique_sorted_x[1])
    
    # b. computing the line equation
    b_1 = (y_1 - y_0) / ((n_points + 1) * (unique_sorted_x[1] - unique_sorted_x[0]))
    a_1 = y_0 / (n_points + 1) - b_1 * unique_sorted_x[0]
    lb = - a_1 / b_1
    
    ### UPPER BOUND ###
    
    # a. accounting for duplicates
    y_last = np.sum(data <= unique_sorted_x[-1])
    y_second_to_last = np.sum(data <= unique_sorted_x[-2])
    
    # b. computing the line equation
    b_n = (y_last - y_second_to_last) / ((n_points + 1) * (unique_sorted_x[-1] - unique_sorted_x[-2]))
    a_n = y_last / (n_points + 1) - b_n * unique_sorted_x[-1]
    ub = (1 - a_n) / b_n
    
    # new array with upper and lower bounds
    sorted_x = np.concatenate((np.array([lb]), sorted_x, np.array([ub])))
    
    y = []
    for new_point in new_points:
        if new_point < lb:
            y.append(0)
        elif new_point > ub:
            y.append(1)
        else:
            index = np.where(sorted_x  <= new_point)[0][-1]
            rank_bounds = (index, index + 1)
            bounds = (sorted_x[index], sorted_x[index + 1])
    
            b = 1 / ((n_points + 1) * (bounds[1] - bounds[0]))
            a = rank_bounds[0] / (n_points + 1) - b * bounds[0]
            y.append(a + b * new_point)
    return np.array(y)


if __name__ == "__main__":
    np.random.seed(10)
    data = np.random.normal(0, 1, size=100)
    new_points = np.array([0.3, 1, -2])
    
    print(smoothed_ecdf(new_points=new_points, data=data))