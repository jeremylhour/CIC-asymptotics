#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate latex from pickle files located in output/raw/

Created on Thu Nov 19 09:56:39 2020

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
    models = ['standard_kernel','standard_xavier','smooth_kernel', 'smooth_ls', 'smooth_xavier']
    metrics_set = ['bias', 'RMSE', 'Coverage rate', 'CI size']
    
    color_dico = dict({.2 : 'LightCyan',
                       .3 : 'LightRed',
                       .5 : 'LightGreen',
                       .8 : 'LightYellow',
                       .9 : 'LightBlue'}) # dico for coloring the blocks
    digits = 3 # rounding
    
    ########## CORE CODE ##########
    counter = 0
    
    with open('output/EXPONENTIAL_simulation.tex', "w") as f:
        for i in lambda_x:
            for j in lambda_z:
                for k in alpha_y:
                    pickle_file = 'output/raw/simulations_B='+str(B)+'_lambda_x='+str(i)+'_lambda_z='+str(j)+'_alpha_y='+str(k)
                    try:
                        result = pickle.load(open(pickle_file+'.p','rb'))
                    except:
                        continue

                    param_line = r' & \multicolumn{'+str(len(metrics_set)*len(result))+'}{c}{'+'$\lambda_X$='+str(i)+', $\lambda_Z$='+str(j)+r', $\alpha_Y$='+str(k)+' -- $b_2+d_2$='+ str(round(1-i/j+1/k, 2))+'}'  
                    param_line += '\\\\'
                    for model in models:
                        counter += 1
                        string = model.replace('_', ' ')
                        item = 'model'
                        sample_line = ' '
                        header = r'\begin{longtable}{l|'
                        for sample_size in result:
                            sample_line += r' & \multicolumn{'+str(len(metrics_set))+'}{c}{'+'n='+str(sample_size)+'}'
                            header += ('c'*len(metrics_set))
                            for metric in metrics_set:
                                string += ' & '+str(round(result.get(sample_size).get(metric).get(model), digits))
                                item += ' & '+metric
                        string += '\\\\'
                        item += '\\\\'
                        sample_line += '\\\\'
                        header += '}'

                        ### WRITING
                        if counter == 1:
                            f.write(header)
                            f.write('\n')
                            f.write(r'\caption{}\\ \n')
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

                        if colours:
                            f.write(r'\rowcolor{'+color_dico.get(i)+'} ')
                        f.write(string)
                        f.write('\n')
                        
                    # Adding bootstrap
                    string = "Bootstrap"
                    for sample_size in result:
                        for metric in metrics_set:
                            if metric == 'Coverage rate':
                                string += ' & '+str(round(result.get(sample_size).get('Bootstrap cov. rate'), digits))
                            else:
                                string += ' & '
                    string += '\\\\'
                    f.write(string)
                    f.write('\n')
                            

        f.write(r'\bottomrule')
        f.write('\n')
        f.write(r'\end{longtable}')
        f.write('\n')