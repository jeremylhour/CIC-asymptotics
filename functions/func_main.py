#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main high-level functions for computing the estimator.
The main function is estimator_unknown_ranks.

Created on Wed Nov 11 12:02:50 2020

@author: jeremylhour
"""
import sys, os
sys.path.append(os.path.join(os.getcwd(),'functions/'))

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


def estimator_unknown_ranks(y, x, z, method="smoothed"):
    """
    estimator_unknown_ranks:
        computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank,
        and corresponding standard error
        
    :param y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
    :param x: np.array of the points to project -- corresponds to outcome of treated group at date 0.
    :param z: np.array of the points for distribution -- corresponds to outcome ot untreated group at date 1.
    :param method: can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
    """
    if method == "smoothed":
        u_hat = smoothed_ecdf(new_points=x, data=z)
    if method == "standard":
        ecdf = ECDF(z)
        u_hat = ecdf(x)
    
    denominateur = kernel_density_estimator(x=np.quantile(y, u_hat), data=y) 
    counterfactual_y = np.quantile(y, u_hat)
    theta = counterfactual_y.mean()    
            
    
    """
    compute_zeta:
        compute vector zeta_i as in out paper,
        similar to Q in Athey and Imbens (2006).
    """
    support = np.linspace(1/len(y), 1, len(y), endpoint=True) # = F_y(Y)
    zeta = []
    for point in support:
        indicator = np.zeros(len(u_hat))
        indicator[np.where(point <= u_hat)] = 1
        inside_integral = -(indicator - u_hat)/denominateur
        zeta.append(inside_integral.mean())
    zeta = np.array(zeta)
    
    
    """
    compute_phi:
        compute vector phi_i as in out paper,
        similar to P in Athey and Imbens (2006).
    """
    support = np.linspace(1/len(z), 1, len(z), endpoint=True) # = F_z(Z)
    phi = []
    for point in support:
        indicator = np.zeros(len(u_hat))
        indicator[np.where(point <= u_hat)] = 1
        inside_integral = (indicator-u_hat)/denominateur
        phi.append(inside_integral.mean())
    phi = np.array(phi)
    
    
    """
    compute_epsilon:
        NOT SURE IT WORKS!!!

    """
    inside_integral = -(counterfactual_y[np.newaxis].T - counterfactual_y)
    epsilon = np.array(inside_integral.mean(axis=1))
    
    
    """
    compute standard error
    """
    se = np.sqrt((zeta**2).mean() + (phi**2).mean() + (epsilon**2).mean())

    return theta, se/np.sqrt(len(y))
            
    