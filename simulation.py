#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file to run simulations

Created on Mon Nov  9 12:07:05 2020

@author: jeremylhour
"""

import sys, os
sys.path.append(os.path.join(os.getcwd(),'functions/'))

import numpy as np
import pandas as pd
import random
import time
import yaml
import math

from statsmodels.distributions.empirical_distribution import ECDF

from func_main import *
from func_ecdf import *
from func_simu import *
from func_kde import *

from scipy.stats import expon, pareto
   

########## SETTING UP FOLDER IF NEEDED ##########
if not os.path.exists('output'):
    os.makedirs('output')


########## LOAD YAML CONFIG ##########
#config_file= os.path.join(os.getcwd(),'config_simulation.yml')
config_file= os.path.join(os.getcwd(),sys.argv[1])

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
    
B = config['nb_simu']

lambda_x = config['lambda_x']
lambda_z = config['lambda_z']
alpha_y = config['alpha_y']

    
print('lambda_x={:.2f} -- lambda_z={:.2f} -- alpha_y={:.2f}'.format(lambda_x, lambda_z, alpha_y))
print('Parameter values give b_2={:.2f}'.format(1-lambda_x/lambda_z))
print('Parameter values give d_2={:.2f}'.format(1/alpha_y))
print('So b_2+d_2={:.2f}'.format(1-lambda_x/lambda_z+1/alpha_y))
print('--- Remember, b_2 + d_2 should be below .5 for Theorem 2 to apply')


##### SAVING TO FILE ###########
outfile = 'output/simulations_B='+str(B)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)
    
f = open(outfile+'.txt', "a")
f.write('\n')
f.write('lambda_x={:.2f} -- lambda_z={:.2f} -- alpha_y={:.2f} \n'.format(lambda_x, lambda_z, alpha_y),)
f.write('Parameter values give b_2={:.2f} \n'.format(1-lambda_x/lambda_z))
f.write('Parameter values give d_2={:.2f} \n'.format(1/alpha_y))
f.write('So b_2+d_2={:.2f} \n'.format(1-lambda_x/lambda_z+1/alpha_y))
f.write('\n')
f.close()


########## CORE CODE ##########
sample_size_set = config['sample_size']
big_results = {}

for sample_size in sample_size_set:
    print('Running {} simulations with sample size {}...'.format(B, sample_size))
    
    f = open(outfile+'.txt', "a")
    f.write('Running {} simulations with sample size {}...'.format(B, sample_size))
    f.close()

    random.seed(999)

    results = np.zeros(shape=(B, 3))
    sigma = np.zeros(shape=(B, 2))

    start_time = time.time()

    for b in range(B):
        sys.stdout.write("\r{0}".format(b))
        sys.stdout.flush()
        
        y, z, x, theta0 =  generate_data(distrib_y = pareto(b=alpha_y, loc=1),
                                         distrib_z = expon(scale=lambda_z),
                                         distrib_x = expon(scale=lambda_x),
                                         size = sample_size)
        # Estimator and S.E.
        theta_smooth, sigma_smooth = estimator_unknown_ranks(y, x, z, method="smoothed")
        theta_standard, sigma_standard = estimator_unknown_ranks(y, x, z, method="standard")
        
        # Collecting results
        results[b,] = [theta0, theta_smooth, theta_standard]
        sigma[b,] = [sigma_smooth, sigma_standard]
    
        # Checking division error
        if math.isinf(sigma_smooth):
            print(' -- error for this iteration')
            results[b,] = [np.nan]*3
            sigma[b,] = [np.nan]*2

    print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")

    ########## POST-PROCESSS ##########
    results = pd.DataFrame(results)
    results.dropna(axis=0, inplace=True)
    
    sigma = pd.DataFrame(sigma)
    sigma.dropna(axis=0, inplace=True)
    
    theta0 = results[0].mean()
    
    y_hat = pd.DataFrame({'smoothed': results[1],
                          'standard': results[2]})
    
    sigma_df = pd.DataFrame({'smoothed': sigma[:,0],
                          'standard': sigma[:,1]})
    
    report = performance_report(y_hat, theta0, n_obs=sample_size, sigma=sigma_df, file=outfile)
    big_results[sample_size] = report
    
    
########## PUTTING TOGETHER A LATEX TABLE ##########
digits = 3
metrics_set = ['bias', 'MAE', 'RMSE', 'Coverage rate', 'Quantile .95']

k=0

f = open(outfile+'.txt', "a")
f.write('\n')
f.write(r'\begin{table}')
f.write('\n')

for model in y_hat.columns:
    k += 1
    string = model
    item = 'model'
    sample_line = ' '
    header = r'\begin{tabular}{l|'
    for sample_size in sample_size_set:
        sample_line = sample_line+ r' & \multicolumn{'+str(len(metrics_set))+'}{c}{'+str(sample_size)+'}'
        header = header + 'c*'+str(len(metrics_set))
        for metric in metrics_set:
            string = string+' & '+str(round(big_results[sample_size][metric][model], digits))
            item = item+' & '+metric
    string = string +'\\\\'
    item = item +'\\\\'
    sample_line = sample_line +'\\\\'
    header = header + '}'
    ### WRITING
    if k == 1:
        f.write(header)
        f.write('\n')
        f.write(r'\toprule')
        f.write('\n')
        f.write(sample_line)
        f.write('\n')
        f.write(item)
        f.write('\n')
        f.write(r'\hline')
        f.write('\n')
    f.write(string)
    f.write('\n')

f.write(r'\bottomrule')
f.write('\n')
f.write(r'\end{tabular}')
f.write('\n')
f.write(r'\end{table}')
f.write('\n')
f.close()
            