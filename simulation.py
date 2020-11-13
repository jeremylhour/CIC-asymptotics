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

from statsmodels.distributions.empirical_distribution import ECDF

from func_main import *
from func_ecdf import *
from func_simu import *
from func_kde import *

from scipy.stats import expon, pareto
   

########## LOAD YAML CONFIG ##########
config_file= os.path.join(os.getcwd(),'config_simulation.yml')

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
    

B = config['nb_simu']
sample_size = config['sample_size']

lambda_x = config['lambda_x']
lambda_z = config['lambda_z']
alpha_y = config['alpha_y']

outfile = 'output/simulations_B='+str(B)+'_n='+str(sample_size)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)+'.txt'
if not os.path.exists('output'):
    os.makedirs('output')
    
print('lambda_x={:.2f} -- lambda_z={:.2f} -- alpha_y={:.2f}'.format(lambda_x, lambda_z, alpha_y))
print('Parameter values give b_2={:.2f}'.format(1-lambda_x/lambda_z))
print('Parameter values give d_2={:.2f}'.format(1/alpha_y))
print('So b_2+d_2={:.2f}'.format(1-lambda_x/lambda_z+1/alpha_y))
print('--- Remember, b_2 + d_2 should be below .5 for Theorem 2 to apply')

print('Running {} simulations with sample size {}...'.format(B, sample_size))

##### SAVING TO FILE ###########
f = open(outfile, "a")
f.write('\n')
f.write('lambda_x={:.2f} -- lambda_z={:.2f} -- alpha_y={:.2f} \n'.format(lambda_x, lambda_z, alpha_y),)
f.write('Parameter values give b_2={:.2f} \n'.format(1-lambda_x/lambda_z))
f.write('Parameter values give d_2={:.2f} \n'.format(1/alpha_y))
f.write('So b_2+d_2={:.2f} \n'.format(1-lambda_x/lambda_z+1/alpha_y))
f.write('\n')
f.write('Running {} simulations with sample size {}...'.format(B, sample_size))
f.write('\n')
f.close()


########## CORE CODE ##########

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

print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")


########## POST-PROCESSS ##########

theta0 = results[:,0].mean()

y_hat = pd.DataFrame({'smoothed': results[:,1],
                      'standard': results[:,2]})
sigma_df = pd.DataFrame({'smoothed': sigma[:,0],
                      'standard': sigma[:,1]})

report = performance_report(y_hat, theta0, sigma=sigma_df, file=outfile)