#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Development scripts for estimation of the denominator

Created on Mon Dec 28 21:32:58 2020

@author: jeremylhour
"""

import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/functions/')

import numpy as np
from scipy.stats import expon

from func_ecdf import *
from func_kde import *
from func_simu import *
from func_main import *


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
    
    ### Estimator of theta
    counterfactual_y = np.quantile(y, u_hat)
    theta_hat = counterfactual_y.mean()    
    
    ### Method for estimation of standard error
    if se_method == "lewbel-schennach":
        u_hat = sorted(u_hat)
        F_inverse = np.quantile(y, u_hat)
        inv_density = (np.delete(F_inverse,0) - np.delete(F_inverse,-1)) / (np.delete(u_hat,0) - np.delete(u_hat,-1))
        u_hat = (np.delete(u_hat,0) + np.delete(u_hat,-1))/2
    elif se_method == "kernel":
        inv_density = 1/kernel_density_estimator(x=np.quantile(y, u_hat), data=y)
    
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



# simulate data
y, z, x =  generate_data(distrib_y = expon(scale=10),
                         distrib_z = expon(scale=1),
                         distrib_x = expon(scale=.2),
                         size = 1000)


estimator_unknown_ranks(y, x, z, method="smoothed", se_method="kernel")
estimator_unknown_ranks(y, x, z, method="smoothed", se_method="lewbel-schennach")


