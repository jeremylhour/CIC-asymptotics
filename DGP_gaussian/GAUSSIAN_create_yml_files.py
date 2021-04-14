#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create YML files for parallel computing,
from a main YML file.

Created on Tue Nov 17 17:13:30 2020

@author: jeremylhour
"""
import os
import yaml

if __name__ == '__main__':
    if not os.path.exists('input_configs_GAUSSIAN'):
        os.makedirs('input_configs_GAUSSIAN')
    
    liste = 'files_list_GAUSSIAN.txt'
    
    ########## INPUT PARAMETERS ##########
    config_file = os.path.join(os.getcwd(),'DGP_gaussian/GAUSSIAN_config.yml')
    
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
        
    nb_simu = config['nb_simu']
    sample_size = config['sample_size']
    mu_x = config['mu_x']
    variance_x = config['variance_x']
    
    print('Creating requested YAML files...')
    
    for i in mu_x:
        for j in variance_x:
            file_name = 'input_configs_GAUSSIAN/mu_x='+str(i)+'_variance_x='+str(j)+'.yml'
            f = open(file_name, "w")
            f.write('nb_simu: {}       # Nb. of simulations, should be a scalar and the same for all files.\n'.format(nb_simu))
            f.write('sample_size: {}    # Sample size, should be an array\n'.format(sample_size))
            f.write('mu_x: {}        # Mean of X\n'.format(i))
            f.write('variance_x: {}         # Variance of X\n'.format(j))
            f.close()
        
            g = open(liste,'a')
            g.write(file_name+'\n')
            g.close()