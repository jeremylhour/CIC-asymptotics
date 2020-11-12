#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:18:06 2020

@author: jeremylhour
"""

import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/functions/')

import numpy as np
from scipy.stats import expon

from func_ecdf import *
from func_kde import *
from func_main import *
from func_simu import *


# simulate data
y, z, x, theta0 =  generate_data(distrib_y = expon(scale=10),
                                 distrib_z = expon(scale=1),
                                 distrib_x = expon(scale=.2),
                                 size = 100)


def compute_se(y, x, z, method="smoothed"):
    """
    compute_se:
        ATTENTION: PAS ADAPTE A DIFFERENTES TAILLES ECHANTILLONS
    """
    u_hat = counterfactual_ranks(points_to_predict=x, points_for_distribution=z, method=method)
    denominateur = kernel_density_estimator(x=np.quantile(y, u_hat), data=y) 
    
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
    counterfactual_y = np.quantile(y, u_hat)
    
    inside_integral = -(counterfactual_y[np.newaxis].T - counterfactual_y)
    epsilon = np.array(inside_integral.mean(axis=1))
    
    """
    compute standard error
    """
    se = np.sqrt((zeta**2).mean() + (phi**2).mean() + (epsilon**2).mean())
    return se / np.sqrt(len(y))
        