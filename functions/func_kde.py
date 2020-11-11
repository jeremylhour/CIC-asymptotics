#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to kernel density estimation.

Created on Tue Nov 10 18:26:34 2020

@author: jeremylhour
"""

import numpy as np


def epanechnikov_kernel(x):
    """
    epanechnikov_kernel:
        Epanechnikov kernel function,
        as suggested in Athey and Imbens (2006).
        
    :param x: np.array
    """
    y = (1-(x**2)/5) * 3/(4*np.sqrt(5))
    y[np.where(abs(x) > np.sqrt(5))] = 0
    return y


def kernel_density_estimator(x, data):
    """
    kernel_density_estimator:
        implement kernel density estimator with Silverman's rule of thumb,
        as suggested in Athey and Imbens (2006).
    
    :param x: new points
    :param data: data to estimate the function
        
    """
    h_silverman = 1.06 * data.std() / (len(data)**(1/5))
    y = (x[np.newaxis].T - data)/h_silverman # Broadcast to an array dimension len(x) * len(data)
    y = epanechnikov_kernel(y)/h_silverman
    return y.mean(axis=1)