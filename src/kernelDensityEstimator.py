"""
Functions related to kernel density estimation.

Created on Tue Nov 10 18:26:34 2020

@author : jeremylhour
"""
import numpy as np
from numba import njit

# ------------------------------------------------------------------------------------
# KERNELS
# ------------------------------------------------------------------------------------
@njit
def epanechnikov_kernel(x):
    """
    epanechnikov_kernel:
        Epanechnikov kernel function, as suggested in Athey and Imbens (2006).
    
    Args:
        x (np.array): points.
    """
    y = (1 - x ** 2 / 5) * 3 / (4 * np.sqrt(5))
    return np.where(np.abs(x) > np.sqrt(5), 0, y)

@njit
def gaussian_kernel(x):
    """
    gaussian_kernel:
        Gaussian kernel function.
    
    Args:
     x (np.array): points.
    """
    return np.exp(- x ** 2 / 2) / np.sqrt(2 * np.pi)

# ------------------------------------------------------------------------------------
# KERNEL DENSITY ESTIMATOR
# ------------------------------------------------------------------------------------
@njit
def kernel_density_estimator(x, data, kernel=gaussian_kernel):
    """
    kernel_density_estimator:
        implement kernel density estimator with Silverman's rule of thumb,
        as suggested in Athey and Imbens (2006).
    
    Args:
        x (np.array): new points.
        data (np.array): data to estimate the function.
        kernel (function): function for the kernel.
    """
    h_silverman = 1.06 * data.std() / (len(data) ** (1 / 5))
    y = (np.expand_dims(x, 1) - data) / h_silverman # Broadcast to an array dimension (len(x), len(data))
    y = kernel(y) / h_silverman
    return np.array([np.mean(y[i]) for i in range(len(y))])

if __name__ == "__main__":
    np.random.seed(10)
    data = np.random.normal(0, 1, size=200)
    x = np.array([-1,0,1, -1, 0, 10])
    
    print(kernel_density_estimator(x, data, kernel=gaussian_kernel))
    print(kernel_density_estimator(x, data, kernel=epanechnikov_kernel))