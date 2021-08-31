#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main functions to generate data and analyze performance.

Also displays an example of simulation for one set of parameters.

Created on Wed Nov 11 12:07:14 2020

@author: jeremylhour
"""
import sys
import numpy as np
from numba import njit
import pandas as pd
import time
import yaml
import pickle
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import expon, pareto

from mainFunctions import estimator_unknown_ranks

# ------------------------------------------------------------------------------------
# UNOBSERVED RANKS
# ------------------------------------------------------------------------------------
def true_theta(distrib_y, distrib_z, distrib_x, size=10000):
    """
    true_theta:
        compute the true value of theta,
        by simulation since analytical formula is not possible.

    @param distrib_y: distribution of Y
    @param distrib_z: distribution of Z
    @param distrib_x: distribution of X
    """
    Q_y = distrib_y.ppf # Quantile function of Y
    F_z = distrib_z.cdf # CDF of Z
    Q_x = distrib_x.ppf # Quantile function of X

    U = np.random.uniform(size=size)
    U_tilde = Q_y(F_z(Q_x(U)))
    theta = np.mean(U_tilde)
    return theta

@njit
def analytical_theta(alpha_y, lambda_z, lambda_x):
    """
    analytical_theta:
        compute the true value of theta,
        using an analytical formula.

    @param alpha_y (float): a positive number
    @param lambda_z (float): a positive number
    @param lambda_x (float): a positive number
    """
    theta = 1/(alpha_y*lambda_x/lambda_z - 1)
    return theta

def generate_data(distrib_y, distrib_z, distrib_x, size=1000):
    """
    generate_data:
        generate data following the specified distributions.
        Should be of class "rv_continuous" from scipy.stats

    @param distrib_y: distribution of Y, instance of rv_continuous
    @param distrib_z: distribution of Z, instance of rv_continuous
    @param distrib_x: distribution of X, instance of rv_continuous
    @param size (int): sample size for each vector
    """
    y = distrib_y.ppf(np.random.uniform(size=size))
    z = distrib_z.ppf(np.random.uniform(size=size))
    x = distrib_x.ppf(np.random.uniform(size=size))
    #theta0 = true_theta(distrib_y=distrib_y, distrib_z=distrib_z, distrib_x=distrib_x, size = 100000)
    return y, z, x


# ------------------------------------------------------------------------------------
# OBSERVED RANKS
# ------------------------------------------------------------------------------------
def true_theta_observed_rank(distrib_y, distrib_u, size=10000):
    """
    true_theta:
        compute the true value of theta,
        by simulation since analytical formula is not possible.

    @param distrib_y: distribution of Y
    @param distrib_u: distribution of U
    @param size (int): sample size
    """
    Q_y = distrib_y.ppf # Quantile function of Y
    Q_u = distrib_u.ppf # Quantile function of U

    U = np.random.uniform(size=size)
    U_tilde = Q_y(Q_u(U))
    theta = np.mean(U_tilde)
    return theta


def generate_data_observed_rank(distrib_y, distrib_u, size=1000):
    """
    generate_data:
        generate data following the specified distributions.
        Should be of class "rv_continuous" from scipy.stats

    @param distrib_y: distribution of Y, instance of rv_continuous
    @param distrib_u: distribution of U, instance of rv_continuous
    @param size (int): sample size for each vector
    """
    y = distrib_y.ppf(np.random.uniform(size=size))
    u = distrib_u.ppf(np.random.uniform(size=size))
    theta0 = true_theta_observed_rank(distrib_y=distrib_y, distrib_u=distrib_u, size=100000)
    return y, u, theta0


# ------------------------------------------------------------------------------------
# PERFORMANCE REPORT
# ------------------------------------------------------------------------------------

def performance_report(y_hat, theta0, n_obs, histograms=True, **kwargs):
    """
    performance_report:
        creates the report for simulations,
        computes bias, MSE, MAE and coverage rate.

    @param y_hat (np.array): B x K np.array of B simulations for K estimators
    @param theta0 (float): scalar, true value of theta
    @param n_obs (int): sample size used during simulations
    @param histograms (bool): whether to draw the histograms
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
    report['CI size'] = 2*norm.ppf(0.975)*sigma.mean(axis=0)

    print('Theta_0: {:.2f}'.format(report.get('theta0')))
    print("Number of simulations: {} \n".format(report.get('n_simu')))
    print("Sample size: {} \n".format(report.get('n_obs')))
    for metric in ['bias', 'MAE', 'RMSE', 'Coverage rate', 'CI size', 'Quantile .95']:
        print(metric+': ')
        for model in y_centered.columns:
            print('- {}: {:.4f}'.format(model, report[metric][model]))
        print('\n')

    ##### WRITING TO FILE #####
    with open(file+'.txt', "a") as f:
        f.write('\n')
        f.write('Theta_0: {:.2f} \n'.format(report['theta0']))
        for metric in ['bias', 'MAE', 'RMSE', 'Coverage rate', 'CI size', 'Quantile .95']:
            f.write(metric+': \n')
            for model in y_centered.columns:
                f.write('- {}: {:.4f} \n'.format(model, report[metric][model]))
            f.write('\n')

    ##### SAVING HISTOGRAM #####
    if histograms:
        num_bins = 50
        for model in y_centered.columns:
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(np.sqrt(n_obs)*y_centered[model], num_bins, density=1)
            norm_fit = norm.pdf(bins, scale=np.sqrt(n_obs)*sigma[model].mean())
            ax.plot(bins, norm_fit, '--')
            ax.set_xlabel(r'$n^{1/2}$ ($\hat \theta$ - $\theta_0$)')
            ax.set_ylabel('Probability density')
            ax.set_title(r'Histogram for model: '+model)
            fig.tight_layout()
            plt.savefig(file+'_n='+str(n_obs)+'_'+model+'.jpg',dpi=(96))

    return report


def latex_table(results, file, models=['standard','smoothed', 'smoothed_lewbel-schennach'], digits=3):
    """
    latex_table:
        outputs a latex table from a list of results
    @param results: list of results based on the format results[sample_size][metric][model]
    @param file (str): name of the output file
    @param models (list of str): list of the models
    @param digits (int): dedines the precision
    """
    metrics_set = ['bias', 'MAE', 'RMSE', 'Coverage rate', 'Quantile .95']
    k=0

    with open(file+'.tex', "a") as f:
        f.write('\n')
        f.write(r'\begin{table}')
        f.write('\n')

        for model in models:
            k += 1
            string = model
            item = 'model'
            sample_line = ' '
            header = r'\begin{tabular}{l|'
            for sample_size in results:
                sample_line = sample_line+ r' & \multicolumn{'+str(len(metrics_set))+'}{c}{'+str(sample_size)+'}'
                header = header + ('c'*len(metrics_set))
                for metric in metrics_set:
                    string = string+' & '+str(round(results[sample_size][metric][model], digits))
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
    return None


if __name__ == '__main__':
    print('THIS IS AN EXAMPLE USING EXPONENTIAL DGP')

    print('='*80)
    print('LOADING THE CONFIG')
    print('='*80)

    CONFIG_FILE = 'DGP_exponential/EXPONENTIAL_example.yml'
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)

    B = config.get('nb_simu')
    lambda_x = config.get('lambda_x')
    lambda_z = config.get('lambda_z')
    alpha_y = config.get('alpha_y')


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
    sample_size_set = config.get('sample_size')
    big_results = {}

    for sample_size in sample_size_set:
        print('Running {} simulations with sample size {}...'.format(B, sample_size))
        with open(outfile+'.txt', "a") as f:
            f.write('Running {} simulations with sample size {}...'.format(B, sample_size))

        np.random.seed(999)
        results, sigma = np.zeros(shape=(B, nb_estimators)), np.zeros(shape=(B, nb_estimators))

        start_time = time.time()
        for b in range(B):
            sys.stdout.write("\r{0}".format(b))
            sys.stdout.flush()

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

            # Collecting results
            results[b,] = [theta_standard, theta_standard_x, theta_smooth, theta_ls, theta_x]
            sigma[b,] = [sigma_standard, sigma_standard_x, sigma_smooth, sigma_ls, sigma_x]

            # Checking division error
            if np.isinf(sigma_smooth) or np.isnan(sigma_ls):
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

        big_results[sample_size] = performance_report(y_hat, theta0, n_obs=sample_size, histograms=False, sigma=sigma_df, file=outfile)


    print('='*80)
    print('SAVING RESULT OBJECT')
    print('='*80)

    pickle_file = 'output/raw/simulations_B='+str(B)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)
    pickle.dump(big_results, open(pickle_file+'.p','wb'))