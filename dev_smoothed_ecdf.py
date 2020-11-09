#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:12:20 2020

@author: jeremylhour
"""
import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/')

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from func import *

import matplotlib.pyplot as plt

# generate uniform points
n = 200
x = np.random.uniform(size=n)
        

support = np.linspace(-0.01, 1.1, 1000, endpoint=False)

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