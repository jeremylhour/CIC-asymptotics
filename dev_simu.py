#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:07:05 2020

@author: jeremylhour
"""
import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/functions/')

import numpy as np
import pandas as pd
import random
import time
from statsmodels.distributions.empirical_distribution import ECDF

from func_main import *
from func_ecdf import *
from func_simu import *
from func_kde import *

from scipy.stats import expon
   


random.seed(999)

B = 500

### Core code

results = np.zeros(shape=(B, 3))
sigma = np.zeros(shape=(B, 2))

start_time = time.time()

for b in range(B):
    sys.stdout.write("\r{0}".format(b))
    sys.stdout.flush()
    
    y, z, x, theta0 =  generate_data(distrib_y = expon(scale=2),
                                     distrib_z = expon(scale=1),
                                     distrib_x = expon(scale=.2),
                                     size = 1000)
    theta_smooth = estimator_unknown_ranks(outcome = y, points_to_translate=x, points_for_distribution=z, method="smoothed")
    theta_standard = estimator_unknown_ranks(outcome = y, points_to_translate=x, points_for_distribution=z, method="standard")
    results[b,] = [theta0, theta_smooth, theta_standard]
    
    sigma_smooth = compute_se(y, x, z, method="smoothed")
    sigma_standard = compute_se(y, x, z, method="standard")
    sigma[b,] = [sigma_smooth, sigma_standard]

print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")

# Post-process
theta0 = results[:,0].mean()
print('Theta_0 vaut: {:.2f}'.format(theta0))

y_hat = pd.DataFrame({'smoothed': results[:,1],
                      'standard': results[:,2]})
sigma_df = pd.DataFrame({'smoothed': sigma[:,0],
                      'standard': sigma[:,1]})

report = performance_report(y_hat, theta0, sigma=sigma_df)