#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file to run simulations with Gaussian DGP.

This DGP :
    - Y ~ N(0,1),
    - Z ~ N(0,1),
    - X ~ N(mu,v).
    
Created on Mon Nov  9 12:07:05 2020

@author: jeremylhour
"""
import sys, os

sys.path.append(os.path.join(os.getcwd(), "functions/"))

import numpy as np
import pandas as pd
import random
import time
import yaml
import math
import pickle

from func_main import estimator_unknown_ranks
from func_simu import generate_data, performance_report, true_theta

from scipy.stats import norm

if __name__ == '__main__':
    print('='*80)
    print('LOADING CONFIG')
    print('='*80)

    config_file = os.path.join(os.getcwd(), sys.argv[1])
    
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    
    B = config["nb_simu"]
    variance_x = config["variance_x"]
    mu_x = config["mu_x"]
    
    
    print(f"mu_x={mu_x}",
          f"variance_x={variance_x}",
          f"Parameter values give b_1={1 - variance_x}",
          f"Parameter values give b_2={1 - variance_x}",
          "--- Remember, b_2 + d_2 should be below .5 for Theorem 2 to apply",
          "--- and below 1 for theta_0 to be finite.",
          sep = '\n'
          )
    
    
    ########## SAVING TO FILE ###########
    outfile = (
        "output/gaussian_simulations_B="
        + str(B)
        + "_mu_x="
        + str(mu_x)
        + "_variance_x="
        + str(variance_x)
    )
    
    with open(outfile + ".txt", "a") as f:
        f.write("\n")
        f.write("mu_x={:.2f} \n".format(mu_x))
        f.write("variance_x={:.2f} \n".format(variance_x))
        f.write("Parameter values give b_1={:.2f} \n".format(1 - variance_x))
        f.write("Parameter values give b_2={:.2f} \n".format(1 - variance_x))
        f.write("\n")
    
    
    print('='*80)
    print('RUNNING SIMULATIONS')
    print('='*80)
    
    nb_estimators = 5
    sample_size_set = config["sample_size"]
    big_results = {}
    
    for sample_size in sample_size_set:
        print("Running {} simulations with sample size {}...".format(B, sample_size))
        with open(outfile + ".txt", "a") as f:
            f.write("Running {} simulations with sample size {}...".format(B, sample_size))
    
        random.seed(999)
    
        results, sigma = np.zeros(shape=(B, nb_estimators)), np.zeros(shape=(B, nb_estimators))
    
        start_time = time.time()
        for b in range(B):
            sys.stdout.write("\r{0}".format(b))
            sys.stdout.flush()
    
            y, z, x = generate_data(
                distrib_y=norm(loc=0, scale=1),
                distrib_z=norm(loc=0, scale=1),
                distrib_x=norm(loc=mu_x, scale=np.sqrt(variance_x)),
                size=sample_size,
            )
            # Estimator and standard error
            theta_standard, sigma_standard = estimator_unknown_ranks(
                y, x, z, method="standard"
            )
            theta_standard_x, sigma_standard_x = estimator_unknown_ranks(
                y, x, z, method="standard", se_method="xavier"
            )
    
            theta_smooth, sigma_smooth = estimator_unknown_ranks(
                y, x, z, method="smoothed", se_method="kernel"
            )
            theta_ls, sigma_ls = estimator_unknown_ranks(
                y, x, z, method="smoothed", se_method="lewbel-schennach"
            )
            theta_x, sigma_x = estimator_unknown_ranks(
                y, x, z, method="smoothed", se_method="xavier"
            )
    
            # Collecting results
            results[b,] = [
                theta_standard,
                theta_standard_x,
                theta_smooth,
                theta_ls,
                theta_x,
            ]
            sigma[b,] = [sigma_standard, sigma_standard_x, sigma_smooth, sigma_ls, sigma_x]
    
            # Checking division error
            if math.isinf(sigma_smooth) or np.isnan(sigma_ls):
                print(" -- error for this iteration")
                results[b,] = [np.nan] * nb_estimators
                sigma[b,] = [np.nan] * nb_estimators
        print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")
    
        ########## POST-PROCESSS ##########
        results = pd.DataFrame(results)
        results.dropna(axis=0, inplace=True)
    
        sigma = pd.DataFrame(sigma)
        sigma.dropna(axis=0, inplace=True)
    
        # theta0 = analytical_theta(alpha_y = alpha_y, lambda_z = lambda_z, lambda_x = lambda_x)
    
        theta0 = true_theta(
            distrib_y=norm(loc=0, scale=1),
            distrib_z=norm(loc=0, scale=1),
            distrib_x=norm(loc=mu_x, scale=np.sqrt(variance_x)),
            size=100000,
        )
    
        y_hat = pd.DataFrame(
            {
                "standard_kernel": results[0],
                "standard_xavier": results[1],
                "smooth_kernel": results[2],
                "smooth_ls": results[3],
                "smooth_xavier": results[4],
            }
        )
    
        sigma_df = pd.DataFrame(
            {
                "standard_kernel": sigma[0],
                "standard_xavier": sigma[1],
                "smooth_kernel": sigma[2],
                "smooth_ls": sigma[3],
                "smooth_xavier": sigma[4],
            }
        )
    
        big_results[sample_size] = performance_report(y_hat, theta0, n_obs=sample_size, histograms=False, sigma=sigma_df, file=outfile)
    
    
    print('='*80)
    print('SAVING RESULT OBJECT')
    print('='*80)
    
    pickle_file = (
        "output/raw/gaussian_simulations_B="
        + str(B)
        + "_mu_x="
        + str(mu_x)
        + "_variance_x="
        + str(variance_x)
    )
    pickle.dump(big_results, open(pickle_file + ".p", "wb"))