#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file to run simulations

Created on Mon Nov  9 12:07:05 2020

@author: jeremylhour
"""

import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/functions/')

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

from scipy.stats import expon
   

########## LOAD YAML CONFIG ##########
config_file='/Users/jeremylhour/Documents/code/CIC-asymptotics/config_simu.yml'

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
    
B = config['nb_simu']
sample_size = config['sample_size']

sample_size = 1000

print('Running {} simulations with sample size {}...'.format(B, sample_size))


########## CORE CODE ##########

random.seed(999)

results = np.zeros(shape=(B, 3))
sigma = np.zeros(shape=(B, 2))

start_time = time.time()

for b in range(B):
    sys.stdout.write("\r{0}".format(b))
    sys.stdout.flush()
    
    y, z, x, theta0 =  generate_data(distrib_y = expon(scale=2),
                                     distrib_z = expon(scale=1),
                                     distrib_x = expon(scale=.2),
                                     size = sample_size)
    # Estimator and true value
    theta_smooth = estimator_unknown_ranks(outcome = y, points_to_translate=x, points_for_distribution=z, method="smoothed")
    theta_standard = estimator_unknown_ranks(outcome = y, points_to_translate=x, points_for_distribution=z, method="standard")
    results[b,] = [theta0, theta_smooth, theta_standard]
    
    # Standard error
    sigma_smooth = compute_se(y, x, z, method="smoothed")
    sigma_standard = compute_se(y, x, z, method="standard")
    sigma[b,] = [sigma_smooth, sigma_standard]

print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")


########## POST-PROCESSS ##########

theta0 = results[:,0].mean()

y_hat = pd.DataFrame({'smoothed': results[:,1],
                      'standard': results[:,2]})
sigma_df = pd.DataFrame({'smoothed': sigma[:,0],
                      'standard': sigma[:,1]})

report = performance_report(y_hat, theta0, sigma=sigma_df)