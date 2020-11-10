#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:26:34 2020

@author: jeremylhour
"""

import numpy as np

def epanechnikov_kernel(x):
    if abs(x) > np.sqrt(5):
        y = 0
    else:
        y = (1-x**2/5) * 3/(4*np.sqrt(5))
    return y

def kernel_density_estimator(x, data):

    h = 1.06 * data.std() / (len(data)**(1/5))
    
    y = [epanechnikov_kernel((i-x)/h) for i in data]/h
    return y.mean()