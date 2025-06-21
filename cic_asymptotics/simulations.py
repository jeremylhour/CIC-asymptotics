"""
Main functions to generate data and analyze performance.

Also displays an example of simulation for one set of parameters.

Created on Wed Nov 11 12:07:14 2020

@author: jeremylhour
"""

import numpy as np
from numba import njit


# ------------------------------------------------------------------------------------
# UNOBSERVED RANKS
# ------------------------------------------------------------------------------------
def true_theta(distrib_y, distrib_z, distrib_x, size=10000):
    """
    true_theta:
        compute the true value of theta,
        by simulation since analytical formula is available.

    @param distrib_y (scipy.stats distrib): distribution of Y
    @param distrib_z (scipy.stats distrib): distribution of Z
    @param distrib_x (scipy.stats distrib): distribution of X
    """
    Q_y = distrib_y.ppf  # Quantile function of Y
    F_z = distrib_z.cdf  # CDF of Z
    Q_x = distrib_x.ppf  # Quantile function of X

    U = np.random.uniform(size=size)
    U_tilde = Q_y(F_z(Q_x(U)))
    return np.mean(U_tilde)


@njit
def analytical_theta(alpha_y, lambda_z, lambda_x):
    """
    analytical_theta:
        compute the true value of theta,
        using an analytical formula.

    @param alpha_y (float): a positive number
    @param lambda_z (float): a positive number
    @param lambda_x (float): a positive number
    """
    return 1 / (alpha_y * lambda_x / lambda_z - 1)


def generate_data(distrib_y, distrib_z, distrib_x, size=1000):
    """
    generate_data:
        generate data following the specified distributions.
        Should be of class "rv_continuous" from scipy.stats

    @param distrib_y (scipy.stats distrib): distribution of Y, instance of rv_continuous
    @param distrib_z (scipy.stats distrib): distribution of Z, instance of rv_continuous
    @param distrib_x (scipy.stats distrib): distribution of X, instance of rv_continuous
    @param size (int): sample size for each vector
    """
    y = distrib_y.ppf(np.random.uniform(size=size))
    z = distrib_z.ppf(np.random.uniform(size=size))
    x = distrib_x.ppf(np.random.uniform(size=size))
    return y, z, x


# ------------------------------------------------------------------------------------
# OBSERVED RANKS
# ------------------------------------------------------------------------------------
def true_theta_observed_rank(distrib_y, distrib_u, size=10000):
    """
    true_theta:
        compute the true value of theta,
        by simulation since analytical formula is not avilable.

    @param distrib_y: distribution of Y
    @param distrib_u: distribution of U
    @param size (int): sample size
    """
    Q_y = distrib_y.ppf  # Quantile function of Y
    Q_u = distrib_u.ppf  # Quantile function of U

    U = np.random.uniform(size=size)
    U_tilde = Q_y(Q_u(U))
    return np.mean(U_tilde)


def generate_data_observed_rank(distrib_y, distrib_u, size=1000):
    """
    generate_data:
        generate data following the specified distributions.
        Should be of class "rv_continuous" from scipy.stats

    @param distrib_y: distribution of Y, instance of rv_continuous
    @param distrib_u: distribution of U, instance of rv_continuous
    @param size (int): sample size for each vector
    """
    y = distrib_y.ppf(np.random.uniform(size=size))
    u = distrib_u.ppf(np.random.uniform(size=size))
    theta0 = true_theta_observed_rank(
        distrib_y=distrib_y, distrib_u=distrib_u, size=100000
    )
    return y, u, theta0
