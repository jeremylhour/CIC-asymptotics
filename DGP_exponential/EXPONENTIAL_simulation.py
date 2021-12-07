#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file to run simulations with exponential DGP.

Parameters :
    - Y ~ Pareto(alpha_y),
    - Z ~ Exponential(lambda_z),
    - X ~ Exponential(lambda_x).

Implied "deep" parameters :
    - b_2 = 1-lambda_x/lambda_z,
    - d_2 = 1/alpha_y,
    - b_1 = d_1 = 0.
    
Created on Mon Nov  9 12:07:05 2020

@author: jeremylhour
"""
import sys
import numpy as np
import pandas as pd
import time
import yaml
import math
import pickle

sys.path.append("src/")

from mainFunctions import estimator_unknown_ranks
from simulations import analytical_theta, generate_data, performance_report 

from scipy.stats import expon, pareto
   
if __name__ == '__main__':
    print('='*80)
    print('LOADING THE CONFIG')
    print('='*80)
    
    CONFIG_FILE = sys.argv[1]
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)
        
    B = config.get('nb_simu', 100)
    lambda_x = config.get('lambda_x', .75)
    lambda_z = config.get('lambda_z', 1)
    alpha_y = config.get('alpha_y', 20)
    sample_size_set = config.get('sample_size', [100, 500])
    
    print(f'lambda_x={lambda_x} -- lambda_z={lambda_z} -- alpha_y={alpha_y}',
          f'Parameter values give b_2={round(1-lambda_x/lambda_z, 2)}',
          f'Parameter values give d_2={round(1/alpha_y, 2)}',
          f'So b_2+d_2={round(1-lambda_x/lambda_z+1/alpha_y, 2)}',
          '--- Remember, b_2 + d_2 should be below .5 for Theorem 2 to apply',
          '--- and below 1 for theta_0 to be finite.',
          sep = '\n'
          )
    
    
    ########## SAVING TO FILE ###########
    outfile = 'output/simulations_B='+str(B)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)
        
    with open(outfile+'.txt', "a") as f:
        f.write('\n')
        f.write('lambda_x={:.2f} -- lambda_z={:.2f} -- alpha_y={:.2f} \n'.format(lambda_x, lambda_z, alpha_y),)
        f.write('Parameter values give b_2={:.2f} \n'.format(1-lambda_x/lambda_z))
        f.write('Parameter values give d_2={:.2f} \n'.format(1/alpha_y))
        f.write('So b_2+d_2={:.2f} \n'.format(1-lambda_x/lambda_z+1/alpha_y))
        f.write('\n')
    
    
    print('='*80)
    print('RUNNING SIMULATIONS')
    print('='*80)
    
    nb_estimators = 5
    big_results = {}
    
    for sample_size in sample_size_set:
        print('Running {} simulations with sample size {}...'.format(B, sample_size))
        with open(outfile+'.txt', "a") as f:
            f.write('Running {} simulations with sample size {}...'.format(B, sample_size))
    
        np.random.seed(999)
        results, sigma = np.empty(shape=(B, nb_estimators)), np.empty(shape=(B, nb_estimators))
        bootstrap_quantiles = np.empty(shape=(B, 2))
    
        start_time = time.time()
        for b in range(B):
            sys.stdout.write("\r{0}".format(b))
            sys.stdout.flush()
            
            # Simulate data
            y, z, x =  generate_data(distrib_y = pareto(b=alpha_y, loc=-1),
                                     distrib_z = expon(scale=1/lambda_z),
                                     distrib_x = expon(scale=1/lambda_x),
                                     size = sample_size)
            
            # Estimator and standard error
            theta_standard, sigma_standard = estimator_unknown_ranks(y, x, z, method="standard")
            theta_standard_x, sigma_standard_x = estimator_unknown_ranks(y, x, z, method="standard", se_method="xavier")
            
            theta_smooth, sigma_smooth = estimator_unknown_ranks(y, x, z, method="smoothed", se_method="kernel")
            theta_ls, sigma_ls = estimator_unknown_ranks(y, x, z, method="smoothed", se_method="lewbel-schennach")
            theta_x, sigma_x = estimator_unknown_ranks(y, x, z, method="smoothed", se_method="xavier")
            
            # Confidence interval by bootstrap
            _, empirical_quantile = estimator_unknown_ranks(y, x, z, method="smoothed", bootstrap_quantile = [.025, .975])
            
            # Collecting results
            results[b,] = [theta_standard, theta_standard_x, theta_smooth, theta_ls, theta_x]
            sigma[b,] = [sigma_standard, sigma_standard_x, sigma_smooth, sigma_ls, sigma_x]
            bootstrap_quantiles[b,] = empirical_quantile
            
            # Checking division error
            if math.isinf(sigma_smooth) or np.isnan(sigma_ls):
                print(' -- error for this iteration')
                results[b,] = [np.nan]*nb_estimators
                sigma[b,] = [np.nan]*nb_estimators
        print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")
    
        ########## POST-PROCESSS ##########
        results = pd.DataFrame(results)
        results.dropna(axis=0, inplace=True)
        
        sigma = pd.DataFrame(sigma)
        sigma.dropna(axis=0, inplace=True)
        
        theta0 = analytical_theta(alpha_y = alpha_y, lambda_z = lambda_z, lambda_x = lambda_x)
        
        y_hat = pd.DataFrame({'standard_kernel': results[0],
                              'standard_xavier': results[1],
                              'smooth_kernel': results[2],
                              'smooth_ls': results[3],
                              'smooth_xavier': results[4]})
        
        sigma_df = pd.DataFrame({'standard_kernel': sigma[0],
                                 'standard_xavier': sigma[1],
                                 'smooth_kernel': sigma[2],
                                 'smooth_ls': sigma[3],
                                 'smooth_xavier': sigma[4]})
        
        big_results[sample_size] = performance_report(y_hat, theta0, n_obs=sample_size, bootstrap_quantiles=bootstrap_quantiles, histograms=False, sigma=sigma_df, file=outfile)
        
    
    print('='*80)
    print('SAVING RESULT OBJECT')
    print('='*80)
    
    pickle_file = 'output/raw/simulations_B='+str(B)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)+'.p'
    pickle.dump(big_results, open(pickle_file,'wb'))