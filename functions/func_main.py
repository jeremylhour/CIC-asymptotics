#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main high-level functions for computing the estimator

Created on Wed Nov 11 12:02:50 2020

@author: jeremylhour
"""
import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/functions/')

from func_kde import *
from func_ecdf import *

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


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
            
    