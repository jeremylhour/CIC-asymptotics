#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:07:05 2020

@author: jeremylhour
"""
import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/')

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from func import *

from scipy.stats import expon
    
### Test
y, z, x, theta0 =  generate_data(expon(scale=.5), expon(scale=1), expon(scale=.7), size = 10000)

estimator_unknown_ranks(outcome = y, points_to_translate=x, points_for_distribution=z, method="smoothed")