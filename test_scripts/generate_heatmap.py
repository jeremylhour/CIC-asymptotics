#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate heatmaps for the coverage rates

Created on Fri Apr 16 20:54:23 2021

@author: jeremylhour
"""
import yaml
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ########## LOAD MAIN CONFIG ##########
    config_file = '../DGP_exponential/EXPONENTIAL_for_heatmap.yml'
    
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
        
    B = config['nb_simu']
    lambda_x = sorted(config['lambda_x'], reverse=True)
    b2 = [round(1 - i, 3) for i in lambda_x]
    
    alpha_y = config['alpha_y']
    d2 = [round(1/i, 3) for i in alpha_y]
    # NB : in this application, lambda_Z = 1
    
    ########## PARAMETERS ##########
    model = 'smooth_xavier'
    metrics = 'Coverage rate'
    sample_size = config['sample_size'][-1]
    
    ########## EXTRACT COVERAGE RATES ##########
    coverageRates = []
    for index_y in alpha_y:
        currentRow = []
        for index_x in lambda_x:
            pickle_file = '../output/raw/simulations_B='+str(B)+'_lambda_x='+str(index_x)+'_lambda_z=1_alpha_y='+str(index_y)+'.p'
            try:
                result = pickle.load(open(pickle_file,'rb'))
                currentRow.append(result[sample_size][metrics][model])
            except:
                currentRow.append(np.nan)
        coverageRates.append(currentRow)
    
    ########## DRAW THE HEATMAP ##########
    ax = sns.heatmap(coverageRates, vmin=0, vmax=1, cmap=sns.color_palette("mako", as_cmap=True),
                     xticklabels=b2, yticklabels=d2)
    ax.set(xlabel=r'$b_2$', ylabel=r'$d_2$')
    plt.savefig('heatmap.png', dpi=600)