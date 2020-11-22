#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate latex from the raw results output.

Created on Thu Nov 19 09:56:39 2020

@author: jeremylhour
"""
import os, sys
import numpy as np
import yaml
import pickle

########## LOAD MAIN CONFIG ##########
config_file = os.path.join(os.getcwd(),'main_config.yml')

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
    
nb_simu = config['nb_simu']
sample_size = config['sample_size']
lambda_x = config['lambda_x']
lambda_z = config['lambda_z']
alpha_y = config['alpha_y']




pickle_file = 'output/raw/simulations_B='+str(B)+'_lambda_x='+str(lambda_x)+'_lambda_z='+str(lambda_z)+'_alpha_y='+str(alpha_y)

favorite_color = pickle.load(open(pickle_file+'.p','rb') )

def latex_table(results, file, models=['standard','smoothed'], digits=3):
    """
    latex_table:
        outputs a latex table from a list of results
    :param results: list of results based on the format results[sample_size][metric][model]
    :param file: name of the output file
    """
    metrics_set = ['bias', 'MAE', 'RMSE', 'Coverage rate', 'Quantile .95']

    k=0

    f = open(file+'.tex', "a")
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
    f.close()