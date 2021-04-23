#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to line charts for the coverage rates,
GAUSSIAN DGP

Created on Fri Apr 16 20:54:23 2021

@author: jeremylhour
"""
import yaml
import pickle
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    ########## LOAD MAIN CONFIG ##########
    config_file = '../DGP_gaussian/GAUSSIAN_config.yml'
    
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
        
    B = config['nb_simu']
    variance_x = sorted(config['variance_x'], reverse=True)
    b2 = [round(1 - 1/i, 2) for i in variance_x]
    
    ########## PARAMETERS ##########
    mu_x = config['mu_x']
    models = ['standard_kernel','standard_xavier','smooth_kernel', 'smooth_ls', 'smooth_xavier']
    sample_size = config['sample_size'][-1]
    
    ########## COLORS AND LINESTYLES ##########
    col_dico = {
            'standard_kernel': 'blue',
            'standard_xavier': 'green',
            'smooth_kernel': 'red',
            'smooth_ls': 'orange',
            'smooth_xavier': 'yellow'
            }
        
    lt_dico = {
            'standard_kernel': 'solid',
            'standard_xavier': 'dashed',
            'smooth_kernel': 'dotted',
            'smooth_ls': 'dashdot',
            'smooth_xavier': (0, (3, 5, 1, 5, 1, 5))
            }
    
    for mean_x in mu_x:
        ########## EXTRACT COVERAGE RATES ##########
        coverageRates = []
    
        for index_x in variance_x:
            pickle_file = '../output/raw/gaussian_simulations_B='+str(B)+'_mu_x='+str(mean_x)+'_variance_x='+str(index_x)+'.p'
            try:
                result = pickle.load(open(pickle_file,'rb'))
                dico = {'bd_sum': 1-1/index_x}
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
        for model in models:
            plt.plot(model, data=dataForPlot, marker='', linestyle=lt_dico[model], markersize=12, color=col_dico[model], linewidth=2)
        
        plt.plot((0, 1), (.95, .95), color='grey', linestyle='dashed')
        plt.plot((.5, .5), (0, 1), color='grey', linestyle='dashed')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.xlabel(r'$b_1, b_2$')
        plt.ylabel('Coverage rate')
        plt.title(r'$\mu_X = $'+str(mean_x))
        plt.show()
            
        #plt.savefig('line_chart.png', dpi=600)