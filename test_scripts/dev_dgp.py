#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:46:53 2021

@author: jeremylhour
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def density(x, delta = 0, sigma0=1, sigma1=1):
    
    # Compute the density
    quantile = norm.ppf(x)
    y = (sigma0/sigma1)*np.exp(-((sigma0+sigma1)*quantile + delta)*((sigma0-sigma1)*quantile + delta)/(2*sigma1**2))
    
    # Compute the bound
    b = (sigma0/sigma1)*(2*(1-x))**(sigma0**2/sigma1**2-1)
    
    return y, b


n = 10000
x_eval = np.linspace(-3, 3, n, endpoint=False)

sigma0 = .2
sigma1 = 1

y, z = density(x_eval, delta=0, sigma0=sigma0, sigma1=sigma1)

print('b2 = '+str(1-sigma0**2/sigma1**2))

if any(y>z):
    print('The function goes over the bound. Problem detected.')

plt.plot(x_eval, y)
plt.plot(x_eval, z)

axes = plt.gca()
axes.set_xlim([.99,1])
axes.set_ylim([0,5000])
plt.show()