"""
Main functions to generate data for simulations

Created on Wed Nov 11 12:07:14 2020

@author: jeremylhour
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import expon, pareto, norm


from .default_values import DEFAULT_SAMPLE_SIZE


# ------------------------------------------------------------------------------------
# UNOBSERVED RANKS
# ------------------------------------------------------------------------------------
def true_theta(distrib_y, distrib_z, distrib_x, size: int = 10_000):
    """
    true_theta:
        compute the true value of theta,
        by simulation since analytical formula is available.

    Args:
        distrib_y (scipy.stats distrib): distribution of Y
        distrib_z (scipy.stats distrib): distribution of Z
        distrib_x (scipy.stats distrib): distribution of X
    """
    Q_y = distrib_y.ppf  # Quantile function of Y
    F_z = distrib_z.cdf  # CDF of Z
    Q_x = distrib_x.ppf  # Quantile function of X

    U = np.random.uniform(size=size)
    U_tilde = Q_y(F_z(Q_x(U)))
    return np.mean(U_tilde)


def analytical_theta(alpha_y, lambda_z, lambda_x):
    """
    analytical_theta:
        compute the true value of theta, using an analytical formula.

    Args:
        alpha_y (float): a positive number
        lambda_z (float): a positive number
        lambda_x (float): a positive number
    """
    return 1 / (alpha_y * lambda_x / lambda_z - 1)


def generate_data(distrib_y, distrib_z, distrib_x, size: int = 1_000):
    """
    generate_data:
        generate data following the specified distributions.
        Should be of class "rv_continuous" from scipy.stats

    Args:
        distrib_y (scipy.stats distrib): distribution of Y, instance of rv_continuous
        distrib_z (scipy.stats distrib): distribution of Z, instance of rv_continuous
        distrib_x (scipy.stats distrib): distribution of X, instance of rv_continuous
        size (int): sample size for each vector
    """
    y = distrib_y.ppf(np.random.uniform(size=size))
    z = distrib_z.ppf(np.random.uniform(size=size))
    x = distrib_x.ppf(np.random.uniform(size=size))
    return y, z, x


# ------------------------------------------------------------------------------------
# OBSERVED RANKS
# ------------------------------------------------------------------------------------
def true_theta_observed_rank(distrib_y, distrib_u, size: int = 10_000):
    """
    true_theta:
        compute the true value of theta,
        by simulation since analytical formula is not avilable.

    Args:
        distrib_y: distribution of Y
        distrib_u: distribution of U
        size (int): sample size
    """
    Q_y = distrib_y.ppf  # Quantile function of Y
    Q_u = distrib_u.ppf  # Quantile function of U

    U = np.random.uniform(size=size)
    U_tilde = Q_y(Q_u(U))
    return np.mean(U_tilde)


def generate_data_observed_rank(distrib_y, distrib_u, size: int = 1000):
    """
    generate_data:
        generate data following the specified distributions.
        Should be of class "rv_continuous" from scipy.stats

    Args:
        distrib_y: distribution of Y, instance of rv_continuous
        distrib_u: distribution of U, instance of rv_continuous
        size (int): sample size for each vector
    """
    y = distrib_y.ppf(np.random.uniform(size=size))
    u = distrib_u.ppf(np.random.uniform(size=size))
    theta0 = true_theta_observed_rank(
        distrib_y=distrib_y, distrib_u=distrib_u, size=100000
    )
    return y, u, theta0


# ------------------------------------------------------------------------------------
# DGP
# ------------------------------------------------------------------------------------
@dataclass
class ExponentialDGP:
    """
    ExponentialDGP

    Args:
        n (int): Sample size for the generated data.
        lambda_x (float): Rate parameter for the covariate X.
        lambda_z (float): Rate parameter for the covariate Z.
        alpha_y (float): Shape parameter for the outcome variable Y.


    Returns:
        tuple: A tuple containing the generated data:
            y (numpy.ndarray): Outcome variable.
            z (numpy.ndarray): Covariate Z.
            x (numpy.ndarray): Covariate X.
    """

    n: int = DEFAULT_SAMPLE_SIZE
    lambda_x: float = 0.8
    lambda_z: float = 1
    alpha_y: float = 10

    def __post_init__(self):
        b_2 = 1 - self.lambda_x / self.lambda_z
        d_2 = 1 / self.alpha_y
        if b_2 + d_2 >= 0.5:
            raise ValueError(
                "b_2 + d_2 should be below 0.5 for Theorem 2 to apply and below 1 for theta_0 to be finite."
            )

        self.theta0 = analytical_theta(
            alpha_y=self.alpha_y, lambda_z=self.lambda_z, lambda_x=self.lambda_x
        )

        self.name = f"exponential_dgp_lambda_x={self.lambda_x}_lambda_z={self.lambda_z}_alpha_y={self.alpha_y}"

    def generate(self):
        y, z, x = generate_data(
            distrib_y=pareto(b=self.alpha_y, loc=-1),
            distrib_z=expon(scale=1 / self.lambda_z),
            distrib_x=expon(scale=1 / self.lambda_x),
            size=self.n,
        )
        return y, z, x


@dataclass
class GaussianDGP:
    """
    GaussianDGP
    """

    n: int = DEFAULT_SAMPLE_SIZE
    mu_x: float = 1
    variance_x: float = 1

    def __post_init__(self):
        b_2 = 1 - 1 / self.variance_x
        d_2 = 0.0
        if b_2 + d_2 >= 0.5:
            raise ValueError(
                "b_2 + d_2 should be below 0.5 for Theorem 2 to apply and below 1 for theta_0 to be finite."
            )

        self.theta0 = self.mu_x
        self.name = f"gaussian_dgp_mu_x={self.mu_x}_variance_x={self.variance_x}"

    def generate(self):
        y, z, x = generate_data(
            distrib_y=norm(loc=0, scale=1),
            distrib_z=norm(loc=0, scale=1),
            distrib_x=norm(loc=self.mu_x, scale=np.sqrt(self.variance_x)),
            size=self.n,
        )
        return y, z, x
