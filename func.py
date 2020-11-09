#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:43:00 2020

All functions necessary to run the program

@author: jeremylhour
"""

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


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


def estimator_known_ranks(outcome, ranks):
    """
    estimator_known_ranks:
        computes the estimator (1), i.e. average of quantiles of the outcome for each rank
        
    :param outcome: np.array of the outcome
    :param ranks: np.array of the ranks
    """
    return np.quantile(outcome, ranks).mean()


def counterfactual_ranks(points_to_predict, points_for_distribution, method="smoothed"):
    """
    counterfactual ranks:
        compute \widehat U the value of the CDF at each element of points_to_predict,
        using the empirical CDF defined by 'points_for_distribution'.
    
    :param points_to_predict: points for wich to get the rank in the distribution
    :param points_for_distribution: points for which to compute the CDF
    :param method: can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
    """
    if method == "smoothed":
        y = smoothed_ecdf(new_points=points_to_predict, x=points_for_distribution)
    if method == "standard":
        ecdf = ECDF(points_for_distribution)
        y = ecdf(points_to_predict)
    return y


def estimator_unknown_ranks(outcome, points_to_translate, points_for_distribution, method="smoothed"):
    """
    estimator_unknown_ranks:
        computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank
        
    :param outcome: np.array of the outcome
    :param ranks: np.array of the ranks
    """
    estimated_ranks = counterfactual_ranks(points_to_translate, points_for_distribution=points_for_distribution, method=method)
    theta = estimator_known_ranks(outcome, estimated_ranks)
    return theta

    
def true_theta(distrib_y, distrib_z, distrib_x, size = 10000):
    """
    true_theta:
        compute the true value of theta,
        by simulation since analytical formula is not possible.
        
    :param distrib_y: distribution of Y
    :param distrib_z: distribution of Z
    :param distrib_x: distribution of X
    """
    Q_y = distrib_y.ppf # Quantile function of Y
    F_z = distrib_z.pdf # CDF of Z
    Q_x = distrib_x.ppf # Quantile function of X
    
    U = np.random.uniform(size=size)
    U_tilde = Q_y(F_z(Q_x(U)))
    theta = U_tilde.mean()
    return theta

def generate_data(distrib_y, distrib_z, distrib_x, size = 1000):
    """
    generate_data:
        generate data following the specified distributions. Using the names in scipy.stats
        
    :param distrib_y: distribution of Y
    :param distrib_z: distribution of Z
    :param distrib_x: distribution of X
    :param size: sample size for each vector
    """
    y = distrib_y.ppf(np.random.uniform(size=size))  
    z = distrib_z.ppf(np.random.uniform(size=size)) 
    x = distrib_x.ppf(np.random.uniform(size=size))   
    theta0 = true_theta(distrib_y=distrib_y, distrib_z=distrib_z, distrib_x=distrib_x, size = 100000)
    
    return y, z, x, theta0
    