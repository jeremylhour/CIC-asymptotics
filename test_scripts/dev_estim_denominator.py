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


def est_denominator(y, x, z, method="smoothed"):
    """
    based on Lewbel and Schennach's paper
        
    :param y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
    :param x: np.array of the points to project -- corresponds to outcome of treated group at date 0.
    :param z: np.array of the points for distribution -- corresponds to outcome ot untreated group at date 1.
    :param method: can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
    """
    if method == "smoothed":
        u_hat = smoothed_ecdf(new_points=x, data=z)
        u_hat = sorted(u_hat)
        
    F_inverse = np.quantile(y, u_hat)
    Q = (np.delete(F_inverse,0) - np.delete(F_inverse,-1)) / (np.delete(u_hat,0) - np.delete(u_hat,-1))
    
    u_hat_midpoint = (np.delete(u_hat,0) + np.delete(u_hat,-1))/2
    
    counterfactual_y = np.quantile(y, u_hat)
    theta_hat = counterfactual_y.mean()    
            
    ### Nécessite de réordonner toutes les données
    
    """
    compute_zeta:
        compute vector zeta_i as in out paper,
        similar to Q in Athey and Imbens (2006).
    """
    support = np.linspace(1/len(y), 1, len(y), endpoint=True) # = F_y(Y)
    zeta = []
    for point in support:
        indicator = np.zeros(len(u_hat_midpoint))
        indicator[np.where(point <= u_hat_midpoint)] = 1
        inside_integral = -(indicator - u_hat_midpoint)*Q
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
        indicator = np.zeros(len(u_hat_midpoint))
        indicator[np.where(point <= u_hat_midpoint)] = 1
        inside_integral = (indicator-u_hat_midpoint)*Q
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


est_denominator(y, x, z, method="smoothed")
estimator_unknown_ranks(y, x, z, method="smoothed")

