#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file for simulation with observed ranks

WARNING : DEPRECATED

Created on Mon Nov 23 09:31:07 2020

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
import pickle

from statsmodels.distributions.empirical_distribution import ECDF

from func_main import *
from func_simu import *

from scipy.stats import beta, pareto
   
if __name__ == '__main__':
    ########## SETTING UP FOLDER IF NEEDED ##########
    for dossier in ['output_obs_ranks', 'output_obs_ranks/raw/']
        if not os.path.exists(dossier):
            os.makedirs(dossier)

    ########## LOAD YAML CONFIG ##########
    #config_file= os.path.join(os.getcwd(),'observed_ranks/obs_rank_config.yml')
    config_file= os.path.join(os.getcwd(),sys.argv[1])

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    B = config['nb_simu']

    alpha_u = config['alpha_u']
    alpha_y = config['alpha_y']


    print('alpha_u={:.2f} -- alpha_y={:.2f}'.format(alpha_u, alpha_y))
    print('Parameter values give b_2={:.2f}'.format(1-alpha_u))
    print('Parameter values give d_2={:.2f}'.format(1/alpha_y))
    print('So b_2+d_2={:.2f}'.format(1-alpha_u+1/alpha_y))
    print('--- Remember, b_2 + d_2 should be below .5 for Theorem 2 to apply')


    ##### SAVING TO FILE ###########
    outfile = 'output_obs_ranks/simulations_B='+str(B)+'_alpha_u='+str(alpha_u)+'_alpha_y='+str(alpha_y)

    f = open(outfile+'.txt', "a")
    f.write('\n')
    print('alpha_u={:.2f} -- alpha_y={:.2f} \n'.format(alpha_u, alpha_y))
    print('Parameter values give b_2={:.2f} \n'.format(1-alpha_u))
    print('Parameter values give d_2={:.2f} \n'.format(1/alpha_y))
    print('So b_2+d_2={:.2f}'.format(1-alpha_u+1/alpha_y))
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

        results = np.zeros(shape=(B, 2))
        sigma = np.zeros(shape=(B, 1))

        start_time = time.time()

        for b in range(B):
            sys.stdout.write("\r{0}".format(b))
            sys.stdout.flush()

            y, u, theta0 =  generate_data_observed_rank(distrib_y = pareto(b=alpha_y, loc=1),
                                                        distrib_u = beta(a=1, b=alpha_u),
                                                        size = sample_size)
            # Estimator and S.E.
            theta_hat, sigma_hat = estimator_known_ranks(y, u)

            # Collecting results
            results[b,] = [theta0, theta_hat]
            sigma[b] = sigma_hat

            # Checking division error
            if math.isinf(sigma_hat):
                print(' -- error for this iteration')
                results[b,] = [np.nan]*2
                sigma[b] = np.nan

        print(f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---")

        ########## POST-PROCESSS ##########
        results = pd.DataFrame(results)
        results.dropna(axis=0, inplace=True)

        sigma = pd.DataFrame(sigma)
        sigma.dropna(axis=0, inplace=True)

        theta0 = results[0].mean()

        y_hat = pd.DataFrame({'standard': results[1]})
        sigma_df = pd.DataFrame({'standard': sigma[0]})

        report = performance_report(y_hat, theta0, n_obs=sample_size, sigma=sigma_df, file=outfile)
        big_results[sample_size] = report


    ########## SAVING RESULTS OBJECT ##########
    pickle_file = 'output_obs_ranks/raw/simulations_B='+str(B)+'_alpha_u='+str(alpha_u)+'_alpha_y='+str(alpha_y)
    pickle.dump(big_results, open(pickle_file+'.p','wb'))