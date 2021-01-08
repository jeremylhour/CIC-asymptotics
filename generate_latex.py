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
    
B = config['nb_simu']

lambda_x = config['lambda_x']
lambda_z = config['lambda_z']
alpha_y = config['alpha_y']


########## CREATING TABLE ##########
digits = 3
models = ['standard_kernel','standard_xavier','smooth_kernel', 'smooth_ls', 'smooth_xavier']
metrics_set = ['bias', 'RMSE', 'Coverage rate', 'CI size']

color_dico = dict({.2 : 'LightCyan',
                   .3 : 'LightRed',
                   .5 : 'LightGreen',
                   .8 : 'LightYellow',
                   .9 : 'LightBlue'})

counter = 0

f = open('output/simulation_table.tex', "w")
f.write('\n')
f.write(r'\begin{table}')
f.write('\n')


for i in lambda_x:
    for j in lambda_z:
        for k in alpha_y:
            pickle_file = 'output/raw/simulations_B='+str(B)+'_lambda_x='+str(i)+'_lambda_z='+str(j)+'_alpha_y='+str(k)
            try:
                result = pickle.load(open(pickle_file+'.p','rb'))
            except:
                continue
            
            param_line = r' & \multicolumn{'+str(len(metrics_set)*len(result))+'}{c}{'+'$\lambda_X$='+str(i)+', $\lambda_Z$='+str(j)+r', $\alpha_Y$='+str(k)+' -- $b_2+d_2$='+ str(round(1-i/j+1/k, 2))+'}'  
            param_line = param_line+'\\\\'
            for model in models:
                counter += 1
                string = model
                item = 'model'
                sample_line = ' '
                header = r'\begin{longtable}{l|'
                for sample_size in result:
                    sample_line = sample_line+ r' & \multicolumn{'+str(len(metrics_set))+'}{c}{'+'n='+str(sample_size)+'}'
                    header = header + ('c'*len(metrics_set))
                    for metric in metrics_set:
                        string = string+' & '+str(round(result[sample_size][metric][model], digits))
                        item = item+' & '+metric
                string = string +'\\\\'
                item = item +'\\\\'
                sample_line = sample_line +'\\\\'
                header = header + '}'
                
                ### WRITING
                if counter == 1:
                    f.write(header)
                    f.write('\n')
                    f.write(r'\caption{}\\')
                    f.write('\n')
                    f.write(r'\toprule')
                    f.write('\n')
                    f.write(sample_line)
                    f.write('\n')
                    f.write(item)
                    f.write('\n')
                    f.write(r'\hline')
                    f.write('\n')
                if model == models[0]:
                    f.write(r'\hline')
                    f.write('\n')
                    f.write(param_line)
                    f.write('\n')
                
                f.write(r'\rowcolor{'+color_dico[i]+'} ')
                f.write(string)
                f.write('\n')

f.write(r'\bottomrule')
f.write('\n')
f.write(r'\end{longtable}')
f.write('\n')
f.close()