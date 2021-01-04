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
from scipy.stats import expon, pareto

from func_ecdf import smoothed_ecdf
from func_simu import analytical_theta, generate_data, performance_report 
from func_kde import kernel_density_estimator, gaussian_kernel, epanechnikov_kernel

import random
import time
import math
import pandas as pd


def estimator_unknown_ranks(y, x, z, method="smoothed", se_method="kernel", remove_duplicates=True):
    """
    estimator_unknown_ranks:
        computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank,
        and corresponding standard error. "lewbel-schennach" implement estimation of s.e. based on Lewbel and Schennach's paper
        
    :param y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
    :param x: np.array of the points to project -- corresponds to outcome of treated group at date 0.
    :param z: np.array of the points for distribution -- corresponds to outcome ot untreated group at date 1.
    :param method: can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
    :param se_method: can be "kernel" or "lewbel-schennach" dependant on the type of method for computing 1/f(F^{-1}(u_hat)).
    :param remove_duplicates: if True remove duplicates when using se_method = "lewbel-schennach"
    
    WARNING: "lewbel-schennach" requires "smoothed" to work (?)
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
        u_hat_count = np.ones(len(u_hat))
        
    elif se_method == "lewbel-schennach":
        u_hat = sorted(u_hat)
        u_hat, u_hat_count = np.unique(u_hat, return_counts=True) # order and remove duplicates
        F_inverse = np.quantile(y, u_hat)
        
        inv_density = (np.delete(F_inverse,0) - np.delete(F_inverse,-1)) / (np.delete(u_hat,0) - np.delete(u_hat,-1))
        u_hat = (np.delete(u_hat,0) + np.delete(u_hat,-1))/2 # replaces u_hat by the average of two consecutive u_hat

        if remove_duplicates:
            u_hat_count = np.ones(len(u_hat))
        else:
            u_hat_count = np.delete(u_hat_count,0)


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
        zeta.append(np.average(inside_integral, weights = u_hat_count))
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
        phi.append(np.average(inside_integral, weights = u_hat_count))
    phi = np.array(phi)
        
    
    """
    compute_epsilon:
        Formula of Athey and Imbens (2006)
    """
    epsilon = theta_hat - counterfactual_y
    
    
    """
    compute standard error
    """
    se = np.sqrt((zeta**2).mean() + (phi**2).mean() + (epsilon**2).mean())

    return theta_hat, se/np.sqrt(len(y))
            


# simulate data
y, z, x =  generate_data(distrib_y = expon(scale=10),
                         distrib_z = expon(scale=1),
                         distrib_x = expon(scale=.2),
                         size = 100)


estimator_unknown_ranks(y, x, z, method="smoothed", se_method="kernel")
estimator_unknown_ranks(y, x, z, method="smoothed", se_method="lewbel-schennach", remove_duplicates=True)
estimator_unknown_ranks(y, x, z, method="smoothed", se_method="lewbel-schennach", remove_duplicates=False)

### Simulations test
random.seed(999)

B = 1000
lambda_x = .8
lambda_z = 1
alpha_y = 8
sample_size=200

results = np.zeros(shape=(B, 3))
sigma = np.zeros(shape=(B, 3))

start_time = time.time()

for b in range(B):
    sys.stdout.write("\r{0}".format(b))
    sys.stdout.flush()
    
    y, z, x =  generate_data(distrib_y = pareto(b=alpha_y, loc=-1),
                                     distrib_z = expon(scale=1/lambda_z),
                                     distrib_x = expon(scale=1/lambda_x),
                                     size = sample_size)
    # Estimator and S.E.
    theta_smooth, sigma_smooth = estimator_unknown_ranks(y, x, z, method="smoothed")
    theta_smooth_2, sigma_ls = estimator_unknown_ranks(y, x, z, method="smoothed", se_method="lewbel-schennach", remove_duplicates=True)
    theta_smooth_3, sigma_ls_duplicates = estimator_unknown_ranks(y, x, z, method="smoothed", se_method="lewbel-schennach", remove_duplicates=False)


    # Collecting results
    results[b,] = [theta_smooth, theta_smooth_2, theta_smooth_3]
    sigma[b,] = [sigma_smooth, sigma_ls, sigma_ls_duplicates]
    
    # Checking division error
    if math.isinf(sigma_smooth) or np.isnan(sigma_ls):
        print(' -- error for this iteration')
        results[b,] = [np.nan]*3
        sigma[b,] = [np.nan]*3

print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")


results = pd.DataFrame(results)
results.dropna(axis=0, inplace=True)
    
sigma = pd.DataFrame(sigma)
sigma.dropna(axis=0, inplace=True)
    
theta0 = analytical_theta(alpha_y = alpha_y, lambda_z = lambda_z, lambda_x = lambda_x)
    
y_hat = pd.DataFrame({'smoothed': results[0],
                      'smoothed_lewbel-schennach': results[1],
                      'smoothed_lewbel-schennach_duplicates': results[2]})
    
sigma_df = pd.DataFrame({'smoothed': sigma[0],
                         'smoothed_lewbel-schennach': sigma[1],
                         'smoothed_lewbel_duplicates': sigma[2]})
    
report = performance_report(y_hat, theta0, n_obs=sample_size, histograms=False, sigma=sigma_df, file='dump.txt')