#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main functions to generate data and analyze performance

Created on Wed Nov 11 12:07:14 2020

@author: jeremylhour
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


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


def performance_report(y_hat, theta0, n_obs, **kwargs):
    """
    performance_report:
        creates the report for simulations,
        computes bias, MSE, MAE and coverage rate.
        
    :param y_hat: B x K np.array of B simulations for K estimators
    :param theta0: scalar, true value of theta
    :param n_obs: sample size usez during simulations.
    """
    sigma = kwargs.get('sigma', np.ones(y_hat.shape))
    file = kwargs.get('file', 'default_output_file')
    
    y_centered = y_hat - theta0
    report = {}
    report['theta0'] = theta0
    report['n_simu'] = len(y_hat)
    report['n_obs']  = n_obs
    report['bias']   = y_centered.mean(axis=0)
    report['MAE']    = abs(y_centered).mean(axis=0)
    report['RMSE']   = y_centered.std(axis=0)
    report['Coverage rate'] = (abs(y_centered/sigma) < norm.ppf(0.975)).mean(axis=0)
    report['Quantile .95'] = (np.sqrt(n_obs)*y_centered).quantile(q=.95, axis=0)
    
    print('Theta_0: {:.2f}'.format(report['theta0']))
    print("Number of simulations: {} \n".format(report['n_simu']))
    print("Sample size: {} \n".format(report['n_obs']))
    for metric in ['bias', 'MAE', 'RMSE', 'Coverage rate', 'Quantile .95']:
        print(metric+': ')
        for model in y_centered.columns:
            print('- {}: {:.4f}'.format(model, report[metric][model]))
        print('\n')
        
    ##### WRITING TO FILE #####
    f = open(file+'.txt', "a")
    f.write('\n')
    f.write('Theta_0: {:.2f} \n'.format(report['theta0']))
    for metric in ['bias', 'MAE', 'RMSE', 'Coverage rate', 'Quantile .95']:
        f.write(metric+': \n ')
        for model in y_centered.columns:
            f.write('- {}: {:.4f} \n'.format(model, report[metric][model]))
        f.write('\n')
    f.close()
    
    ##### SAVING HISTOGRAM #####
    num_bins = report['n_simu']//5
    for model in y_centered.columns:
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(np.sqrt(n_obs)*y_centered[model], num_bins, density=1)
        y = (1 / (np.sqrt(2 * np.pi) * sigma[model].mean())) * np.exp(-0.5 * (1 / sigma[model].mean() * bins**2))
        ax.plot(bins, y, '--')
        ax.set_xlabel(r'$n^{1/2}$ ($\hat \theta$ - $\theta_0$)')
        ax.set_ylabel('Probability density')
        ax.set_title(r'Histogram for model: '+model)
        fig.tight_layout()
        plt.savefig(file+'_'+model+'.jpg',dpi=(96))
    
    return report