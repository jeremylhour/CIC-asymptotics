#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main functions to generate data and analyze performance

Created on Wed Nov 11 12:07:14 2020

@author: jeremylhour
"""

import numpy as np
from scipy.stats import norm


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
    F_z = distrib_z.cdf # CDF of Z
    Q_x = distrib_x.ppf # Quantile function of X
    
    U = np.random.uniform(size=size)
    U_tilde = Q_y(F_z(Q_x(U)))
    theta = U_tilde.mean()
    return theta


def generate_data(distrib_y, distrib_z, distrib_x, size = 1000):
    """
    generate_data:
        generate data following the specified distributions.
        Should be of class "rv_continuous" from scipy.stats
        
    :param distrib_y: distribution of Y, instance of rv_continuous
    :param distrib_z: distribution of Z, instance of rv_continuous
    :param distrib_x: distribution of X, instance of rv_continuous
    :param size: sample size for each vector
    """        
    y = distrib_y.ppf(np.random.uniform(size=size))  
    z = distrib_z.ppf(np.random.uniform(size=size)) 
    x = distrib_x.ppf(np.random.uniform(size=size))   
    theta0 = true_theta(distrib_y=distrib_y, distrib_z=distrib_z, distrib_x=distrib_x, size = 100000)
    
    return y, z, x, theta0


def performance_report(y_hat, theta0, **kwargs):
    """
    performance_report:
        creates the report for simulations,
        computes bias, MSE, MAE and coverage rate.
        
    :param y_hat: B x K np.array of B simulations for K estimators
    :param theta0: scalar, true value of theta
    """
    sigma = kwargs.get('sigma', np.ones(y_hat.shape))
    
    y_centered = y_hat - theta0
    report = {}
    report['theta0'] = theta0
    report['n_obs']  = len(y_hat)
    report['bias']   = y_centered.mean(axis=0)
    report['MAE']    = abs(y_centered).mean(axis=0)
    report['RMSE']   = y_centered.std(axis=0)
    report['Coverage rate'] = (abs(y_centered/sigma) < norm.ppf(0.975)).mean(axis=0)
    
    print('Theta_0 vaut: {:.2f}'.format(report['theta0']))
    print("Number of simulations: {} \n".format(report['n_obs']))
    for metric in ['bias', 'MAE', 'RMSE', 'Coverage rate']:
        print(metric+': ')
        for model in y_centered.columns:
            print('- {}: {:.4f}'.format(model, report[metric][model]))
        print('\n')
    
    return report