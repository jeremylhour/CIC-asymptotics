#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to studi the behavior of zeta (eta in the paper)

Created on Mon Nov 29 14:20:10 2021

@author: jeremylhour
"""
import numpy as np
import sys
import yaml
import time
import pickle

from scipy.stats import expon, pareto

from src.simulations import generate_data
from src.mainFunctions import compute_zeta, inv_density_Xavier, counterfactual_ranks


def theoretical_zeta(y, lambda_x, alpha_y):
    """
    theoretical_zeta :
        computes the theoretical value of zeta, for exponential DGP
        
    WARNING : requires lambda_x - 1/alpha_y < 1
        
    @param y (np.array):
    @param lambda_x (float):
    @param alpha_y (float):
    """
    if lambda_x - 1/alpha_y > 1:
        raise ValueError("lambda_x and alpha_y args must be so that lambda_x - 1/alpha_y < 1 for intergal to converge.")
    
    quantile_y = 1-(1+y)**(-alpha_y) # quantile of Y, F_y(Y)
    beta_const = lambda_x - 1/alpha_y - 2
    
    zeta = -lambda_x * ((1-quantile_y)**(beta_const+1) + 1/(beta_const+2)) / (alpha_y*(beta_const+1))
    return zeta


if __name__ == '__main__':    
    print('='*80)
    print('LOADING THE CONFIG')
    print('='*80)
    
    CONFIG_FILE = sys.argv[1]
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)
        
    B = config.get('nb_simu')
    lambda_x = config.get('lambda_x')
    lambda_z = config.get('lambda_z')
    alpha_y = config.get('alpha_y')
    sample_size_set = config.get('sample_size')
    
    print(f'lambda_x={lambda_x} -- lambda_z={lambda_z} -- alpha_y={alpha_y}',
          f'Parameter values give b_2={round(1-lambda_x/lambda_z, 2)}',
          f'Parameter values give d_2={round(1/alpha_y, 2)}',
          f'So b_2+d_2={round(1-lambda_x/lambda_z+1/alpha_y, 2)}',
          '--- Remember, b_2 + d_2 should be below .5 for Theorem 2 to apply',
          '--- and below 1 for theta_0 to be finite.',
          sep = '\n'
          )
    
    
    ########## SAVING TO FILE ###########
    outfile = 'output_zeta/simulations_B='+str(B)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)
        
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
    
    nb_metrics = 3
    big_results = {}
    
    for sample_size in sample_size_set:
        print('Running {} simulations with sample size {}...'.format(B, sample_size))
        with open(outfile+'.txt', "a") as f:
            f.write('Running {} simulations with sample size {}...'.format(B, sample_size))
    
        np.random.seed(999)
        results = np.zeros(shape=(B, nb_metrics))
    
        start_time = time.time()
        for b in range(B):
            sys.stdout.write("\r{0}".format(b))
            sys.stdout.flush()
            
            # SIMULATE DATA
            y, z, x =  generate_data(distrib_y = pareto(b=alpha_y, loc=-1),
                                     distrib_z = expon(scale=1/lambda_z),
                                     distrib_x = expon(scale=1/lambda_x),
                                     size = sample_size)
            
            # COMPUTE THE TRUE ZETA
            zeta = theoretical_zeta(y=np.sort(y), lambda_x=lambda_x, alpha_y=alpha_y)
            
            # COMPUTE THE ESTIMATED ZETA
            u_hat = counterfactual_ranks(points_to_predict=x, points_for_distribution=z)
            inv_density = inv_density_Xavier(u_hat=u_hat, y=y, spacing_2=True)
            zeta_hat = compute_zeta(u_hat, inv_density, size=sample_size)
            
            # COLLECTING RESULTS
            results[b] = [
                np.mean(zeta_hat)-np.mean(zeta),
                np.mean(zeta_hat**2)-np.mean(zeta**2),
                (np.abs(zeta-zeta_hat)).max()
                ]
        
        print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")
    
        ########## POST-PROCESSS ##########
        dico_results = {
            'sample size': sample_size,
            'lambda_x': lambda_x,
            'alpha_y': alpha_y,
            'mean of errors': results[:,0].mean(),
            'mean of errors squared': results[:,1].mean(),
            'max of absolute errors': results[:,2].mean()
            }
        
        big_results[sample_size] = dico_results
        
    print('='*80)
    print('SAVING RESULT OBJECT')
    print('='*80)
    
    pickle_file = 'output_zeta/raw/simulations_B='+str(B)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)+'.p'
    pickle.dump(big_results, open(pickle_file,'wb'))