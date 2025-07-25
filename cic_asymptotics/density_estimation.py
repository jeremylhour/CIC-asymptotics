"""
All function related to density estimation.
Coded in numba for speed.

Created on Tue Nov 10 18:26:34 2020

@author : jeremylhour
"""

import numpy as np
from numba import njit


# ------------------------------------------------------------------------------------
# INVERSE DENSITY ESTIMATORS
# ------------------------------------------------------------------------------------
@njit
def inv_density_ls(u_hat: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """
    inv_density_ls :
        returns u_hat and inv_density using the Lewbel-Schennach (2007) method.
        Also changes the u_hat by removing duplicates

    Source :
        "A Simple Ordered Data Estimator For Inverse Density Weighted Functions,"
        Arthur Lewbel and Susanne Schennach, Journal of Econometrics, 2007, 186, 189-211.

    Args:
        u_hat (np.array): output of counterfactual_ranks function.
        y (np.array): outcome.

    Return:
        u_hat, inv_density (np.array)
        Needs to return the u_hat in case of non-uniqueness.
    """
    u_hat = np.unique(u_hat)  # remove duplicates and order
    F_inverse = np.quantile(y, u_hat)
    inv_density = (F_inverse[1:] - F_inverse[:-1]) / (u_hat[1:] - u_hat[:-1])
    u_hat = 0.5 * (
        u_hat[1:] + u_hat[:-1]
    )  # replaces u_hat by the average of two consecutive u_hat
    return u_hat, inv_density


@njit
def inv_density_xavier(
    u_hat: np.ndarray, y: np.ndarray, spacing_2: bool = True
) -> np.ndarray:
    """
    inv_density_xavier :
        returns inv_density using the Xavier method

    Args:
        u_hat (np.array): output of counterfactual_ranks function
        y (np.array): outcome
        spacing_2 (bool): If True, two spacings between data points as in the points used for U_i will be
            U_i+ and U_i-. If not, it's U_i+ and U_i

    Return:
        inv_density (np.array)
    """
    u_hat_sorted = np.sort(u_hat)
    unique_u = np.unique(u_hat_sorted)

    # Build maps of next higher and next lower unique values
    idx_upper = np.searchsorted(unique_u, u_hat_sorted, side="right")
    idx_lower = np.searchsorted(unique_u, u_hat_sorted, side="left") - 1

    ub = np.where(idx_upper >= len(unique_u), u_hat_sorted, unique_u[idx_upper])

    if spacing_2:
        lb = np.where(idx_lower < 0, u_hat_sorted, unique_u[idx_lower])
    else:
        # Use u itself unless it's max or spacing_2 is True
        is_max = u_hat_sorted == np.max(u_hat_sorted)
        lb = np.where(is_max, unique_u[idx_lower], u_hat_sorted)

    inv_density = (np.quantile(y, ub) - np.quantile(y, lb)) / (ub - lb)
    return inv_density


# ------------------------------------------------------------------------------------
# KERNELS
# ------------------------------------------------------------------------------------
@njit
def rectangular_kernel(x: np.ndarray) -> np.ndarray:
    """
    rectangular_kernel:
        Rectangular kernel function.

    Args:
        x (np.array): points.
    """
    return np.where(np.abs(x) <= 1, 0.5, 0.0)


@njit
def epanechnikov_kernel(x: np.ndarray) -> np.ndarray:
    """
    epanechnikov_kernel:
        Epanechnikov kernel function, as suggested in Athey and Imbens (2006).

    Args:
        x (np.array): data points.

    Returns:
        np.array: values of the Epanechnikov kernel at points x.
    """
    return np.where(np.abs(x) > np.sqrt(5), 0, (1 - x**2 / 5) * 3 / (4 * np.sqrt(5)))


@njit
def gaussian_kernel(x: np.ndarray) -> np.ndarray:
    """
    gaussian_kernel:
        Gaussian kernel function.

    Args:
     x (np.array): points.

    Returns:
        np.array: values of the Gaussian kernel at points x.
    """
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


@njit
def cosine_kernel(x: np.ndarray) -> np.ndarray:
    """
    cosine_kernel:
        Cosine kernel function.

    Args:
        x (np.array): points.

    Returns:
        np.array: values of the Cosine kernel at points x.
    """
    return np.where(np.abs(x) <= 1, 0.25 * np.pi * np.cos(0.5 * np.pi * x), 0.0)


@njit
def silverman_kernel(x: np.ndarray) -> np.ndarray:
    """
    silverman_kernel:
        Silverman's kernel function

    Args:
        x (np.array): points.

    Returns:
        np.array: values of the Silverman's kernel at points x.
    """
    return (
        0.5
        * np.exp(-np.abs(x) / np.sqrt(2))
        * np.sin(np.abs(x) / np.sqrt(2) + 0.25 * np.pi)
    )


# ------------------------------------------------------------------------------------
# KERNEL DENSITY ESTIMATORS
# ------------------------------------------------------------------------------------
@njit
def kernel_density_estimator(
    x: np.ndarray, data: np.ndarray, kernel=gaussian_kernel
) -> np.ndarray:
    """
    kernel_density_estimator:
        implements kernel density estimator with Silverman's rule of thumb,
        as suggested in Athey and Imbens (2006).

    Args:
        x (np.array): new points at which to evaluate the density.
        data (np.array): data to estimate the function.
        kernel (function): kernel.

    Returns:
        result (np.array): estimated density at points x.
    """
    n = len(data)
    h = 1.06 * np.std(data) / (n**0.2)  # Silverman's rule of thumb
    m = len(x)
    result = np.empty(m)

    for i in range(m):
        s = 0.0
        for j in range(n):
            s += kernel((x[i] - data[j]) / h)
        result[i] = s / (n * h)
    return result


@njit
def kernel_density_estimator_with_moving_bandwith(
    x: np.ndarray, data: np.ndarray
) -> np.ndarray:
    """
    kernel_density_estimator_with_moving_bandwith:
        Implements a kernel density estimator with a moving bandwith.

    Args:
        x (np.array): new points at which to evaluate the density.
        data (np.array): data to estimate the function.

    Returns:
        result (np.array): estimated density at points x.
    """
    n = len(data)
    m = len(x)
    result = np.empty(m)

    for i in range(m):
        xi = x[i]
        hi = xi * (1.0 - xi) / np.log(n)
        count = 0
        for j in range(n):
            if abs(xi - data[j]) <= hi:
                count += 1
        result[i] = count / (2 * hi * n)
    return result
