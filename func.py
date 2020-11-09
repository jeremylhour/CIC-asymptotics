#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:43:00 2020

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
        as in Shorack and Wellner (p. 86)
    
    :param new_points: new points for which the value is returned
    :param x: points used to compute the smoothed empirical CDF
    """
    n_points = len(x)
    sorted_x = np.zeros(n_points+1)
    sorted_x[1:] =  np.array(sorted(x))
    sorted_x = np.append(sorted_x,1) # add 0 first and 1 at the end
    
    y = []
    for new_point in new_points:
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
    
    estimated_ranks = counterfactual_ranks(points_to_translate, points_for_distribution=points_for_distribution, method=method)
    theta = estimator_known_ranks(outcome, estimated_ranks)
    return theta
    
    