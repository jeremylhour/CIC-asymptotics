#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to estimation of empirical CDF,
mainly smoothed empirical CDF as in Shorack and Wellner

Created on Mon Nov  9 13:43:00 2020

@author: jeremylhour
"""

import numpy as np


def ranks_and_antiranks(points):
    """
    ranks_and_antiranks:
        returns ranks and antiranks for array of points
        
    :param points: np.array, vector of points of dimension 1
    """
    ranks = np.array([sorted(points).index(i) for i in points])
    return ranks, np.argsort(ranks)


def smoothed_ecdf(new_points, x):
    """
    smoothed_ecdf:
        Smoothed empirical CDF
        as in Shorack and Wellner (p. 86), but extended to non-bounded support.
        Linear extension outside the support using the nearest linear parts.
    
    :param new_points: new points for which the value is returned
    :param x: points used to compute the smoothed empirical CDF
    """
    n_points = len(x)
    sorted_x = np.array(sorted(x))
    
    ### Compute extreme values outside the support (cf. email Xavier 09/11),
    ### by extending the affine smoothing to the origin or to 1.
    # Lower bound
    b_1 = 1/((n_points+1)*(sorted_x[1]-sorted_x[0]))
    a_1 = 1/(n_points+1) - b_1*sorted_x[0]
    lb = -a_1/b_1
    
    # Upper bound
    b_n = 1/((n_points+1)*(sorted_x[-1]-sorted_x[-2]))
    a_n = (n_points-1)/(n_points+1) - b_n*sorted_x[-2]
    ub = (1-a_n)/b_n
    
    sorted_x = np.insert(sorted_x, 0, lb) # add lb first
    sorted_x = np.append(sorted_x, ub) # add ub last
    
    y = []
    for new_point in new_points:
        if new_point < lb:
            y.append(0)
        elif new_point> ub:
            y.append(1)
        else:
            index = np.where(sorted_x  <= new_point)[0][-1]
            rank_bounds = [index, index+1]
            bounds = sorted_x[rank_bounds]
    
            b = 1/((n_points+1)*(bounds[1]-bounds[0]))
            a = rank_bounds[0]/(n_points+1) - b*bounds[0]
            y.append(a + b*new_point)
        
    return np.array(y)