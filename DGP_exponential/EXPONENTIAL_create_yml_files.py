#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create YAML files for parallel computing, from a main YAML file.
Exponential DGP

Created on Tue Nov 17 17:13:30 2020

@author: jeremylhour
"""
import os
import yaml

if __name__ == '__main__':
    print("="*80)
    print('LOADING CONFIG')
    print("="*80)
    
    if not os.path.exists('input_configs_EXPONENTIAL'):
        os.makedirs('input_configs_EXPONENTIAL')
    JOB_FILE = 'job_list.txt'
    
    CONFIG_FILE ='DGP_exponential/EXPONENTIAL_config.yml'
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)
        
    # Allocate params
    nb_simu = config.get('nb_simu')
    sample_size = config.get('sample_size')
    lambda_x = config.get('lambda_x')
    lambda_z = config.get('lambda_z')
    alpha_y = config.get('alpha_y')
    
    print("="*80)
    print('CREATING REQUESTED YAML FILES')
    print("="*80)
    
    for i in lambda_x:
        for j in lambda_z:
            for k in alpha_y:
                if(1-i/j + 1/k < 1):
                    file_name = 'input_configs_EXPONENTIAL/lambda_x='+str(i)+'_lambda_z='+str(j)+'_alpha_y='+str(k)+'.yml'
                    with open(file_name, "w") as f:
                        f.write('nb_simu: {}       # Nb. of simulations, should be a scalar and the same for all files.\n'.format(nb_simu))
                        f.write('sample_size: {}    # Sample size, should be an array\n'.format(sample_size))
                        f.write('lambda_x: {}        # Parameter of exponential distribution for X\n'.format(i))
                        f.write('lambda_z: {}         # Parameter of exponential distribution for Z\n'.format(j))
                        f.write('alpha_y: {}         # Parameter of Pareto distribution for Y\n'.format(k))

                    with open(JOB_FILE, 'a') as g:
                        g.write(file_name+'\n')
                else:
                    print('Bad DGP : lambda_x='+str(i)+', lambda_z='+str(j)+', alpha_y='+str(k)+' -- true value theta_0 is not finite.')