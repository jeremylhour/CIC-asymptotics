#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for developing analytical formulas for theta_0

Created on Thu Nov 26 11:44:16 2020

@author: jeremylhour
"""

import numpy as np
import random
from func_main import *
from func_ecdf import *
from func_simu import *
from func_kde import *

from scipy.stats import expon, pareto



########## PARAM ##########
lambda_z = 1.5
lambda_x = .9
alpha_y = 3

print('lambda_x={:.2f} -- lambda_z={:.2f} -- alpha_y={:.2f}'.format(lambda_x, lambda_z, alpha_y))
print('Parameter values give b_2={:.2f}'.format(1-lambda_x/lambda_z))
print('Parameter values give d_2={:.2f}'.format(1/alpha_y))
print('So b_2+d_2={:.2f}'.format(1-lambda_x/lambda_z+1/alpha_y))

if lambda_z > alpha_y*lambda_x:
    print('Analytical formula is not working in that case.')

########## SIMULATED ##########
theta_sim = true_theta(distrib_y = pareto(b=alpha_y, loc=-1),
                       distrib_z = expon(scale=1/lambda_z),
                       distrib_x = expon(scale=1/lambda_x),
                       size = 100000)

print('Theta, value computed by Monte Carlo: {:.3f}'.format(theta_sim))

########## ANALYTICAL ##########

theta_an = analytical_theta(alpha_y = alpha_y,
                            lambda_z = lambda_z,
                            lambda_x = lambda_x)

print('Theta, value computed analytically: {:.3f}'.format(theta_an))


########## CHECK FOR WHAT VALUES TO CHOOSE ###########
lambda_z = 1

lambda_x = np.array([.2, .3, .5, .8, .9])
b_2 = 1-lambda_x 
alpha_y = np.array([1.5, 2, 3, 4, 6, 7, 10])
d_2 = 1/alpha_y

print(b_2+d_2[np.newaxis].T)
