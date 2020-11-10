#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:24:50 2020

@author: jeremylhour
"""

import sys
sys.path.append('/Users/jeremylhour/Documents/code/CIC-asymptotics/')

import numpy as np
from func_sd import *

n = 100
data = norm.ppf(np.random.uniform(size=n))


support = np.linspace(-5, 5, 1000, endpoint=False)

y_1 = [kernel_density_estimator(point, data) for point in support]

fig, ax = plt.subplots(1, 1)
ax.plot(support, y_1, '-k', lw=2, label='Empirical Density')
ax.set_xlabel('Support')
ax.set_ylabel('Cumulative probability')
ax.legend(fancybox=True, loc='right')

plt.show()