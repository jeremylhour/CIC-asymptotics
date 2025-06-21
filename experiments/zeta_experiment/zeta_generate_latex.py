#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate latex from pickle files located in output_zeta/raw/

Created on Wed Dec 1 22:19:39 2021

@author: jeremylhour
"""
import yaml
import pickle

if __name__ == '__main__':
    ########## LOAD MAIN CONFIG ##########
    CONFIG_FILE = 'DGP_exponential/EXPONENTIAL_config.yml'
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)
        
    B = config.get('nb_simu')
    lambda_x = config.get('lambda_x')
    lambda_z = config.get('lambda_z')
    alpha_y = config.get('alpha_y')
    colours = False
    
    ########## CREATING TABLE ##########
    print("="*80)
    print('CREATING LaTeX TABLE')
    print("="*80)
    
    ########## PARAMETERS ##########
    metrics_set = ['mean of errors', 'mean of errors squared', 'max of absolute errors']
    formula = {
        'mean of errors': '$E[\hat \eta_i - \eta_i]$',
        'mean of errors squared': '$E[\hat \eta_i^2 - \eta_i^2]$',
        'max of absolute errors': '$\max E[\vert \hat \eta_i - \eta_i \vert$'
               }
    
    color_dico = dict({.2 : 'LightCyan',
                       .3 : 'LightRed',
                       .5 : 'LightGreen',
                       .8 : 'LightYellow',
                       .9 : 'LightBlue'}) # dico for coloring the blocks
    digits = 3 # rounding
    
    ########## CORE CODE ##########
    counter = 0
    
    with open('output_zeta/EXPONENTIAL_simulation.tex', "w") as f:
        for i in lambda_x:
            for j in lambda_z:
                for k in alpha_y:
                    pickle_file = 'output_zeta/raw/simulations_B='+str(B)+'_lambda_x='+str(i)+'_lambda_z='+str(j)+'_alpha_y='+str(k)
                    try:
                        result = pickle.load(open(pickle_file+'.p','rb'))
                    except:
                        continue

                    param_line = r' & \multicolumn{'+str(len(metrics_set)*len(result))+'}{c}{'+'$\lambda_X$='+str(i)+', $\lambda_Z$='+str(j)+r', $\alpha_Y$='+str(k)+' -- $b_2+d_2$='+ str(round(1-i/j+1/k, 2))+'}'  
                    param_line = param_line+'\\\\'

                    counter += 1
                    string = ''
                    item = 'model'
                    sample_line = ' '
                    header = r'\begin{longtable}{l|'
                    for sample_size in result:
                        sample_line = sample_line+ r' & \multicolumn{'+str(len(metrics_set))+'}{c}{'+'n='+str(sample_size)+'}'
                        header = header + ('c'*len(metrics_set))
                        for metric in metrics_set:
                            string = string+' & '+str(round(result[sample_size][metric], digits))
                            item = item+' & '+formula[metric]
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
                        
                    f.write(r'\hline')
                    f.write('\n')
                    f.write(param_line)
                    f.write('\n')

                    if colours:
                        f.write(r'\rowcolor{'+color_dico[i]+'} ')
                    f.write(string)
                    f.write('\n')

        f.write(r'\bottomrule')
        f.write('\n')
        f.write(r'\end{longtable}')
        f.write('\n')