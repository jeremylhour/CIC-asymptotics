#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create YML files for parallel computing,
from a main YML file.

Created on Tue Nov 17 17:13:30 2020

@author: jeremylhour
"""

import sys, os
import yaml

if not os.path.exists('input_configs'):
    os.makedirs('input_configs')

liste = 'files_list.txt'

########## INPUT PARAMETERS ##########
config_file = os.path.join(os.getcwd(),'main_config.yml')

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
    
nb_simu = config['nb_simu']
sample_size = config['sample_size']
lambda_x = config['lambda_x']
lambda_z = config['lambda_z']
alpha_y = config['alpha_y']


for i in lambda_x:
    for j in lambda_z:
        for k in alpha_y:
            if(1-i/j + 1/k < 1):
                file_name = 'input_configs/lambda_x='+str(i)+'_lambda_z='+str(j)+'_alpha_y='+str(k)+'.yml'
                f = open(file_name, "w")
                f.write('nb_simu: {}       # Nb. of simulations, should be a scalar and the same for all files.\n'.format(nb_simu))
                f.write('sample_size: {}    # Sample size, should be an array\n'.format(sample_size))
                f.write('lambda_x: {}        # Parameter of exponential distribution for X\n'.format(i))
                f.write('lambda_z: {}         # Parameter of exponential distribution for Z\n'.format(j))
                f.write('alpha_y: {}         # Parameter of Pareto distribution for Y\n'.format(k))
                f.close()
            
                g = open(liste,'a')
                g.write(file_name+'\n')
                g.close()
            else:
                print('Will not consider DGP with lambda_x='+str(i)+', lambda_z='+str(j)+', and alpha_y='+str(k)+' since true value is not finite.')