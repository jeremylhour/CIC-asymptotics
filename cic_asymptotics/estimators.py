"""
Main high-level functions for computing the estimator.
The main function is estimator_unknown_ranks.

Created on Wed Nov 11 12:02:50 2020

@author : jeremylhour
"""

import numpy as np
from numba import njit

from statsmodels.distributions.empirical_distribution import ECDF

from .empirical_cdf import smoothed_empirical_cdf
from .density_estimation import (
    kernel_density_estimator,
    inv_density_ls,
    inv_density_xavier,
)


# ------------------------------------------------------------------------------------
# LOWER-LEVEL FUNCTIONS
# ------------------------------------------------------------------------------------
@njit
def compute_zeta(u_hat: np.ndarray, inv_density: np.ndarray, size: int) -> np.ndarray:
    """
    compute_zeta :
        function to compute zeta, similar to Q in Athey and Imbens (2006).
        In our paper, it might be called 'eta' instead.

    Args:
        u_hat (np.array): output of counterfactual_ranks function.
        inv_density (np.array): output of any of the inv_density functions.
        size (int): size of the support.

    Return:
        zeta (np.array)
    """
    zeta = np.empty(size)
    for i in range(size):
        s = (i + 1) / size  # support point
        acc = 0.0
        for j in range(len(u_hat)):
            indicator = 1.0 if s <= u_hat[j] else 0.0
            acc += -(indicator - u_hat[j]) * inv_density[j]
        zeta[i] = acc / len(u_hat)
    return zeta


@njit
def compute_phi(u_hat: np.ndarray, inv_density: np.ndarray, size: int) -> np.ndarray:
    """
    compute_phi :
        function to compute phi, similar to P in Athey and Imbens (2006).
        Same function as compute_zeta, but with a different sign.

    Args:
        u_hat (np.array): output of counterfactual_ranks function
        inv_density (np.array): output of any of the inv_density functions
        size (int): size of the support

    Return:
        phi (np.array)
    """
    phi = np.empty(size)
    for i in range(size):
        s = (i + 1) / size
        acc = 0.0
        for j in range(len(u_hat)):
            indicator = 1.0 if s <= u_hat[j] else 0.0
            acc += (indicator - u_hat[j]) * inv_density[j]
        phi[i] = acc / len(u_hat)
    return phi


@njit
def compute_theta(u_hat: np.ndarray, y: np.ndarray) -> tuple:
    """
    compute_theta :
        returns theta_hat and counterfactual outcome

    Args:
        u_hat (np.array): output of counterfactual_ranks function
        y (np.array): outcome

    Return:
        counterfactual_y, theta_hat (np.array)
    """
    counterfactual_y = np.quantile(y, u_hat)
    theta_hat = np.mean(counterfactual_y)
    return counterfactual_y, theta_hat


def counterfactual_ranks(
    x: np.ndarray, points_for_distribution: np.ndarray, method: str = "smoothed"
) -> np.ndarray:
    """
    counterfactual_ranks :
        compute \widehat U the value of the CDF at each element of points_to_predict,
        using the empirical CDF defined by 'points_for_distribution' (i.e. Z in the paper).

    Args:
        x (np.array): points for wich to get the rank in the distribution
        points_for_distribution (np.array): points for which to compute the CDF
        method (str): can be "smoothed" or "standard" dependant on the type of method for computation of the CDF

    Return:
        u_hat (np.array)
    """
    if method == "smoothed":
        u_hat = smoothed_empirical_cdf(x=x, data=points_for_distribution)
    elif method == "standard":
        ecdf = ECDF(points_for_distribution)
        u_hat = ecdf(x)
    else:
        raise ValueError(
            "'method' argument for counterfactual ranks needs to be either 'smoothed' or 'standard'."
        )
    return u_hat


# ------------------------------------------------------------------------------------
# UNKNOWN RANKS
# ------------------------------------------------------------------------------------
def estimator_unknown_ranks(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    method: str = "smoothed",
    se_method: str = "kernel",
    bootstrap_quantile=None,
):
    """
    estimator_unknown_ranks :
        computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank,
        and corresponding standard error. "lewbel-schennach" implement estimation of s.e. based on Lewbel and Schennach's paper

    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome ot untreated group at date 1.
        method (str): can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
        se_method (str): can be "kernel", "lewbel-schennach", or "xavier" depending on the type of method for computing 1/f(F^{-1}(u_hat)).
        bootstrap_quantile (list): Bootstrap quantiles to compute, if None, skip this

    Return:
        estimator and standard errors (np.array)
    """
    u_hat = counterfactual_ranks(x=x, points_for_distribution=z, method=method)

    """
    Estimator of theta
    """
    counterfactual_y, theta_hat = compute_theta(u_hat=u_hat, y=y)

    """
    Compute quantiles and stop, if needed.
    """
    if bootstrap_quantile:
        theta_bootstrap = compute_bootstrap_sample(y, x, z, method=method)
        return theta_hat, np.quantile(theta_bootstrap, q=bootstrap_quantile)

    """
    Compute inv_density depending on the method of choice.
    """
    if se_method == "kernel":
        inv_density = 1.0 / kernel_density_estimator(x=np.quantile(y, u_hat), data=y)
    elif se_method == "lewbel-schennach":
        u_hat, inv_density = inv_density_ls(u_hat=u_hat, y=y)
    elif se_method == "xavier":
        inv_density = inv_density_xavier(u_hat=u_hat, y=y)
    else:
        raise ValueError(
            "'se_method' argument should be 'kernel', 'lewbel-schennach' or 'xavier'."
        )

    """
    compute_zeta:
        compute vector zeta_i as in out paper,
        similar to Q in Athey and Imbens (2006).
    """
    zeta = compute_zeta(u_hat=u_hat, inv_density=inv_density, size=len(y))

    """
    compute_phi:
        compute vector phi_i as in out paper,
        similar to P in Athey and Imbens (2006).
    """
    phi = compute_phi(u_hat=u_hat, inv_density=inv_density, size=len(z))

    """
    compute_epsilon:
        Formula of Athey and Imbens (2006)
    """
    epsilon = theta_hat - counterfactual_y

    """
    compute standard error
    """
    se_ = np.sqrt(np.mean(zeta**2) + np.mean(phi**2) + np.mean(epsilon**2)) / np.sqrt(
        len(y)
    )

    return theta_hat, se_


def compute_double_integral():
    """
    compute_double_integral :
    """


# ------------------------------------------------------------------------------------
# KNOWN RANKS
# ------------------------------------------------------------------------------------
def estimator_known_ranks(y, u):
    """
    estimator_known_ranks :
        computes the estimator (1), i.e. average of quantiles of the outcome for each rank

    WARNING :
        This function is a particular case of the previous one,
        it can probably be included with the right options, but I haven't took the time to do it.

    Args:
        y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
        u: np.array of the ranks
    """
    inv_density = 1 / kernel_density_estimator(x=np.quantile(y, u), data=y)
    counterfactual_y, theta_hat = compute_theta(u_hat=u, y=y)

    """
    compute_zeta:
        compute vector zeta_i as in out paper,
        similar to Q in Athey and Imbens (2006).
    """
    zeta = compute_zeta(u_hat=u, inv_density=inv_density, size=len(y))

    """
    compute_epsilon:
        Formula of Athey and Imbens (2006)
    """
    epsilon = theta_hat - counterfactual_y

    """
    compute standard error
    """
    se = np.sqrt(np.mean(zeta**2) + np.mean(epsilon**2))

    return theta_hat, se / np.sqrt(len(y))


# ------------------------------------------------------------------------------------
# BOOTSTRAP PROCEDURE
# ------------------------------------------------------------------------------------
def compute_bootstrap_sample(y, x, z, B: int = 1_000, method: str = "smoothed"):
    """
    compute_bootstrap_sample :
        computes the bootstrapped sample

    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome of untreated group at date 1.
        B (int): number of bootstrap samples
        method (str): can be "smoothed" or "standard" depending on the type of method for computation of the CDF

    Return:
        (np.array)
    """
    if method not in ["smoothed", "standard"]:
        raise ValueError(
            "'method' argument for counterfactual ranks needs to be either 'smoothed' or 'standard'."
        )

    rng = np.random.default_rng()
    n_x, n_y, n_z = len(x), len(y), len(z)

    results = np.empty(B)
    for b in range(B):
        x_sample = rng.choice(x, size=n_x, replace=True)
        z_sample = rng.choice(z, size=n_z, replace=True)
        y_sample = rng.choice(y, size=n_y, replace=True)
        u_hat = counterfactual_ranks(
            x=x_sample, points_for_distribution=np.unique(z_sample), method=method
        )
        _, theta_hat = compute_theta(u_hat=u_hat, y=y_sample)
        results[b] = theta_hat

    return results
