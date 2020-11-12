#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:24:50 2020

@author: jeremylhour
"""

import sys, os
sys.path.append(os.path.join(os.getcwd(),'functions/'))

import numpy as np
from func_kde import *

from scipy.stats import norm
import matplotlib.pyplot as plt

n = 1000
data = norm.ppf(np.random.uniform(size=n))


support = np.linspace(-5, 5, 1000, endpoint=False)

y_1 = kernel_density_estimator(support, data)
y_2 = norm.pdf(support)

fig, ax = plt.subplots(1, 1)
ax.plot(support, y_1, '-k', lw=2, label='Empirical Density')
ax.plot(support, y_2, '-r', lw=2, label='True Density')
ax.set_xlabel('Support')
ax.set_ylabel('Cumulative probability')
ax.legend(fancybox=True, loc='right')

plt.show()