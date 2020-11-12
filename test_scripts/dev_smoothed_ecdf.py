#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:12:20 2020

@author: jeremylhour
"""
import sys, os
sys.path.append(os.path.join(os.getcwd(),'functions/'))

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from func_ecdf import *

import matplotlib.pyplot as plt
from scipy.stats import powerlaw, norm

# generate uniform points
n = 1000
x = norm.ppf(np.random.uniform(size=n))
        

support = np.linspace(-5, 5, 1000, endpoint=False)

# 1. Non smoothed empirical CDF
ecdf = ECDF(x)
y_1 = ecdf(support)


# 2. Smoothed empirical CDF
y_2 = smoothed_ecdf(support, x)


# 3. Plot
fig, ax = plt.subplots(1, 1)
ax.plot(support, y_1, '-k', lw=2, label='Empirical CDF')
ax.plot(support, y_2, '-r', lw=2, label='Smoothed Empirical CDF')
ax.set_xlabel('Support')
ax.set_ylabel('Cumulative probability')
ax.legend(fancybox=True, loc='right')

plt.show()