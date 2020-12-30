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
        u_hat = smoothed_ecdf(new_points=points_to_predict, x=points_for_distribution)
    if method == "standard":
        ecdf = ECDF(points_for_distribution)
        u_hat = ecdf(points_to_predict)
    return u_hat


def estimator_unknown_ranks(y, x, z, method="smoothed", se_method="kernel"):
    """
    estimator_unknown_ranks:
        computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank,
        and corresponding standard error. "lewbel-schennach" implement estimation of s.e. based on Lewbel and Schennach's paper
        
    :param y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
    :param x: np.array of the points to project -- corresponds to outcome of treated group at date 0.
    :param z: np.array of the points for distribution -- corresponds to outcome ot untreated group at date 1.
    :param method: can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
    :param se_method: can be "kernel" or "lewbel-schennach" dependant on the type of method for computing 1/f(F^{-1}(u_hat)).
    
    WARNING: "lewbel-schennach" requires "smoothed" to work
    """
    if method == "smoothed":
        u_hat = smoothed_ecdf(new_points=x, data=z)
    if method == "standard":
        ecdf = ECDF(z)
        u_hat = ecdf(x)
        
    """
    Estimator of theta
    """
    counterfactual_y = np.quantile(y, u_hat)
    theta_hat = counterfactual_y.mean() 
    
    """
    Computes inv_density depending on the method of choice.
    """
    if se_method == "kernel":
        inv_density = 1/kernel_density_estimator(x=np.quantile(y, u_hat), data=y)
        
    elif se_method == "lewbel-schennach":
        u_hat = sorted(u_hat)
        F_inverse = np.quantile(y, u_hat)
        inv_density = (np.delete(F_inverse,0) - np.delete(F_inverse,-1)) / (np.delete(u_hat,0) - np.delete(u_hat,-1))
        u_hat = (np.delete(u_hat,0) + np.delete(u_hat,-1))/2 # replaces u_hat by the average of two sorted


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
        inside_integral = -(indicator - u_hat)*inv_density
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
        inside_integral = (indicator-u_hat)*inv_density
        phi.append(inside_integral.mean())
    phi = np.array(phi)
    
    
    
    """
    compute_epsilon:
        Formula of Athey and Imbens (2006)
    """
    epsilon = -(counterfactual_y - theta_hat)
    
    
    """
    compute standard error
    """
    se = np.sqrt((zeta**2).mean() + (phi**2).mean() + (epsilon**2).mean())

    return theta_hat, se/np.sqrt(len(y))
            

def estimator_known_ranks(y, u):
    """
    estimator_known_ranks:
        computes the estimator (1), i.e. average of quantiles of the outcome for each rank
        
    :param y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
    :param u: np.array of the ranks
    
    VERIFIER QUE C'EST BON!!!
    """
    
    denominateur = kernel_density_estimator(x=np.quantile(y, u), data=y) 
    counterfactual_y = np.quantile(y, u)
    theta_hat = counterfactual_y.mean()    
            
    
    """
    compute_zeta:
        compute vector zeta_i as in out paper,
        similar to Q in Athey and Imbens (2006).
    """
    support = np.linspace(1/len(y), 1, len(y), endpoint=True) # = F_y(Y)
    zeta = []
    for point in support:
        indicator = np.zeros(len(u))
        indicator[np.where(point <= u)] = 1
        inside_integral = -(indicator - u)/denominateur
        zeta.append(inside_integral.mean())
    zeta = np.array(zeta)
    
    
    """
    compute_epsilon:
        Formula of Athey and Imbens (2006)
    """
    epsilon = -(counterfactual_y - theta_hat)
    
    
    """
    compute standard error
    """
    se = np.sqrt((zeta**2).mean() + (epsilon**2).mean())
    
    return theta_hat, se/np.sqrt(len(y))