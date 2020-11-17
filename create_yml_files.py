#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create YML files for parallel computing

Created on Tue Nov 17 17:13:30 2020

@author: jeremylhour
"""

import sys, os
import yaml

if not os.path.exists('input_configs'):
    os.makedirs('input_configs')

liste = 'files_list.txt'

########## INPUT PARAMETERS ##########
nb_simu = 1000        # Nb. of simulations, should be a scalar and the same for all files.
sample_size = [50, 100, 200, 500, 1000, 10000]    # Sample size, should be an array


lambda_x = [.4, .5, .6, .7, .8]        # Parameter of exponential distribution for X
lambda_z = [1]         # Parameter of exponential distribution for Z
alpha_y  = [2,3,4,5,8]          # Parameter of Pareto distribution for Y


for i in lambda_x:
    for j in lambda_z:
        for k in alpha_y:
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