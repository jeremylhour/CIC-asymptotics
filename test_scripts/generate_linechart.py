#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to line charts for the coverage rates

Created on Fri Apr 16 20:54:23 2021

@author: jeremylhour
"""
import yaml
import pickle
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    ########## LOAD MAIN CONFIG ##########
    config_file = '../DGP_exponential/EXPONENTIAL_config.yml'
    
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
        
    B = config['nb_simu']
    lambda_x = sorted(config['lambda_x'], reverse=True)
    b2 = [round(1 - i, 2) for i in lambda_x]
    
    alpha_y = sorted(config['alpha_y'], reverse=False)
    d2 = [round(1/i, 2) for i in alpha_y]
    # NB : in this application, lambda_Z = 1
    
    ########## PARAMETERS ##########
    models = ['standard_kernel','standard_xavier','smooth_kernel', 'smooth_ls', 'smooth_xavier']
    sample_size = config['sample_size'][-1]
    
    ########## EXTRACT COVERAGE RATES ##########
    coverageRates = []
    for index_y in alpha_y:
        currentRow = []
        for index_x in lambda_x:
            pickle_file = '../output/raw/simulations_B='+str(B)+'_lambda_x='+str(index_x)+'_lambda_z=1_alpha_y='+str(index_y)+'.p'
            try:
                result = pickle.load(open(pickle_file,'rb'))
                dico = {'bd_sum': 1-index_x + 1/index_y}
                for model in models:
                    dico[model] = result[sample_size]['Coverage rate'][model]
                coverageRates.append(dico)
            except:
                pass
    
    ########## AVERAGE OVER VALUES OF b_2+d_2 ##########
    df = pd.DataFrame(coverageRates)
    df['bd_sum'] = round(df['bd_sum'], 2) # pour gérer les approximations
    dataForPlot = df.groupby('bd_sum').mean()
    
    ########## DRAW THE CHART ##########
    col_dico = {
        'standard_kernel': 'blue',
        'standard_xavier': 'green',
        'smooth_kernel': 'red',
        'smooth_ls': 'orange',
        'smooth_xavier': 'yellow'
        }
    
    for model in models:
        plt.plot(model, data=dataForPlot, marker='', linestyle='solid', markersize=12, color=col_dico[model], linewidth=2)
    
    plt.plot((0, 1), (.95, .95), color='grey', linestyle='dashed')
    plt.plot((.5, .5), (0, 1), color='grey', linestyle='dashed')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel(r'$b_2+d_2$')
    plt.ylabel('Coverage rate')
        
    plt.savefig('line_chart.png', dpi=600)