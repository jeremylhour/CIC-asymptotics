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



def analytical_theta(lambda_x, alpha_y):
    return 1/(alpha_y*lambda_x - 1)


########## PARAM ##########
lambda_z = 1
lambda_x = .9
alpha_y = 3

print('lambda_x={:.2f} -- lambda_z={:.2f} -- alpha_y={:.2f}'.format(lambda_x, lambda_z, alpha_y))
print('Parameter values give b_2={:.2f}'.format(1-lambda_x/lambda_z))
print('Parameter values give d_2={:.2f}'.format(1/alpha_y))
print('So b_2+d_2={:.2f}'.format(1-lambda_x/lambda_z+1/alpha_y))

########## SIMULATED ##########
theta_sim = true_theta(distrib_y = pareto(b=alpha_y, loc=-1),
                       distrib_z = expon(scale=1/lambda_z),
                       distrib_x = expon(scale=1/lambda_x),
                       size = 100000)

print(theta_sim)

########## ANALYTICAL ##########

theta_an = analytical_theta(lambda_x = lambda_x,
                            alpha_y = alpha_y)

print(theta_an)




lambda_x = np.array([.2, .3, .5, .8, .9])
b_2 = 1-lambda_x
alpha_y = np.array([1, 2, 4, 6, 8])
d_2 = 1/alpha_y

b_2+d_2[np.newaxis].T