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
    kernel_density_estimator_with_moving_bandwith,
)


# ------------------------------------------------------------------------------------
# LOWER-LEVEL FUNCTIONS
# ------------------------------------------------------------------------------------
@njit
def compute_theta_hat(
    u_hat: np.ndarray, y: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """
    compute_theta_hat :
        returns theta_hat and counterfactual outcome

    Args:
        u_hat (np.array): output of counterfactual_ranks function
        y (np.array): outcome

    Return:
        counterfactual_y, theta_hat (np.array)
    """
    counterfactual_y = np.quantile(y, u_hat)
    theta_hat = np.mean(counterfactual_y)
    return theta_hat, counterfactual_y


def compute_counterfactual_ranks(
    x: np.ndarray, data: np.ndarray, method: str = "smoothed"
) -> np.ndarray:
    """
    counterfactual_ranks :
        compute \widehat U the value of the CDF at each element of points_to_predict,
        using the empirical CDF defined by 'data' (i.e. Z in the paper).

    Args:
        x (np.array): points for wich to get the rank in the distribution
        data (np.array): points for which to compute the CDF
        method (str): can be "smoothed" or "standard" dependant on the type of method for computation of the CDF

    Return:
        u_hat (np.array)
    """
    if method == "smoothed":
        return smoothed_empirical_cdf(x=x, data=data)
    elif method == "standard":
        ecdf = ECDF(data)
        return ecdf(x)
    else:
        raise ValueError(
            "'method' argument for counterfactual ranks needs to be either 'smoothed' or 'standard'."
        )


@njit
def compute_zeta(u_hat: np.ndarray, inv_density: np.ndarray, size: int) -> np.ndarray:
    """
    compute_zeta :
        function to compute zeta, similar to Q in Athey and Imbens (2006).
        In our paper, it might be called 'eta' instead.

        Notice also that this function allows to compute phi, similar to P in Athey and Imbens (2006).
        Same function as compute_zeta, but with a different sign, i.e. phi_i = -zeta_i.

    Args:
        u_hat (np.array): output of counterfactual_ranks function.
        inv_density (np.array): output of any of the inv_density functions.
        size (int): size of the support.

    Return:
        zeta (np.array), i.e. -1 * phi
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
def w(x: float, y: float) -> float:
    """
    w :
        function to compute w(x, y) = min(x, y) * min(1 - x, 1 - y)

    Args:
        x (float): first point
        y (float): second point
    """
    return np.minimum(x, y) * np.minimum(1 - x, 1 - y)


@njit
def compute_double_integral(y: np.array, x: np.array, z: np.array) -> float:
    """
    compute_double_integral :
        computes the double integral of the product of two kernel density estimators
        with a moving bandwith, as in the paper.

    Args:
        y (np.array): outcome
        x (np.array): points to project
        z (np.array): points for distribution
    Returns:
        (float): the value of the double integral
    """
    n_y, n_x, n_z = len(y), len(x), len(z)
    N = min(n_y, n_z, n_x)

    y_ = 2 * np.arange(1, n_y // 2) / n_y

    u_hat1 = smoothed_empirical_cdf(x=x[: n_x // 2], data=z[: n_z // 2])
    u_hat2 = smoothed_empirical_cdf(x=x[n_x // 2 :], data=z[n_z // 2 :])

    f_1 = kernel_density_estimator_with_moving_bandwith(y_, u_hat1)
    f_2 = kernel_density_estimator_with_moving_bandwith(y_, u_hat2)

    interval_1 = np.diff(np.sort(y[: n_y // 2]))
    interval_2 = np.diff(np.sort(y[n_y // 2 :]))

    s = 0.0
    for i in range(n_y // 2 - 1):
        for j in range(n_y // 2 - 1):
            s += f_1[i] * f_2[j] * w(y_[i], y_[j]) * interval_1[i] * interval_2[j]

    return s * N * (n_y + n_z) / (n_y * n_z)


# ------------------------------------------------------------------------------------
# ESTIMATOR (OF THETA, PARAMETER OF INTEREST)
# ------------------------------------------------------------------------------------
def compute_estimator(
    y: np.ndarray,
    x: np.ndarray = None,
    z: np.ndarray = None,
    u: np.ndarray = None,
    cdf_method: str = "smoothed",
) -> "tuple[np.ndarray, np.ndarray]":
    """
    compute_estimator :
        computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank.
        If z is not None, it computes the counterfactual ranks based on z.
        If u is not None, it computes the counterfactual ranks based on u.
        If both z and u are None, it raises an error.

    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome ot untreated group at date 1.
        u (np.array): the points for which the ranks are known.
        method (str): can be "smoothed" or "standard" depending on the type of method of estimation for the CDF.

    Return:
        theta_hat, counterfactual_y (np.array)
        and u_hat if (x, z) is not None.
    """
    if (z is not None) and (x is not None) and (u is None):
        # Compute counterfactual ranks based on z
        u_hat = compute_counterfactual_ranks(x=x, data=z, method=cdf_method)
        theta_hat, counterfactual_y = compute_theta_hat(u_hat=u_hat, y=y)
        return theta_hat, counterfactual_y, u_hat
    elif (u is not None) and (z is None) and (x is None):
        theta_hat, counterfactual_y = compute_theta_hat(u_hat=u_hat, y=y)
        return theta_hat, counterfactual_y
    else:
        raise ValueError("Either (x, z) or u must be provided, but not both.")


# ------------------------------------------------------------------------------------
# COMPUTE STANDARD ERROR -- ESTIMATED RANKS CASE
# ------------------------------------------------------------------------------------
def compute_standard_error_estimated_ranks(
    y: np.ndarray,
    x: np.ndarray = None,
    z: np.ndarray = None,
    cdf_method: str = "smoothed",
    se_method: str = "sample-splitting",
) -> np.ndarray:
    """
    compute_standard_error_estimated_ranks :
            computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank,
            and corresponding standard error. "lewbel-schennach" implement estimation of s.e. based on Lewbel and Schennach's paper

    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome ot untreated group at date 1.
        cdf_method (str): can be "smoothed" or "standard" dependant on the type of method for computation of the CDF.
        se_method (str): can be "sample-splitting", "kernel", "lewbel-schennach", or "xavier" depending on the type of method for computing 1/f(F^{-1}(u_hat)).
    """
    # Compute the estimator with unknown ranks
    theta_hat, counterfactual_y, u_hat = compute_estimator(
        y=y, x=x, z=z, cdf_method=cdf_method
    )

    # Compute standard error
    epsilon = theta_hat - counterfactual_y

    if se_method == "sample-splitting":
        n_y, n_x, n_z = len(y), len(x), len(z)
        N = min(n_y, n_z, n_x)

        part_i = compute_double_integral(y=y, x=x, z=z)
        part_ii = N * np.mean(epsilon**2) / len(x)

        standard_error = np.sqrt(part_i + part_ii) / np.sqrt(N)

    elif se_method in ["kernel", "lewbel-schennach", "xavier"]:
        if se_method == "kernel":
            inv_density = 1.0 / kernel_density_estimator(
                x=np.quantile(y, u_hat), data=y
            )
        elif se_method == "lewbel-schennach":
            u_hat, inv_density = inv_density_ls(u_hat=u_hat, y=y)
        elif se_method == "xavier":
            inv_density = inv_density_xavier(u_hat=u_hat, y=y)

        zeta = compute_zeta(u_hat=u_hat, inv_density=inv_density, size=len(y))
        phi = -compute_zeta(u_hat=u_hat, inv_density=inv_density, size=len(z))

        standard_error = np.sqrt(
            np.mean(zeta**2) + np.mean(phi**2) + np.mean(epsilon**2)
        ) / np.sqrt(len(y))
    else:
        raise ValueError(
            "'se_method' argument should be 'sample-splitting', 'kernel', 'lewbel-schennach' or 'xavier'."
        )

    return theta_hat, standard_error


# ------------------------------------------------------------------------------------
# COMPUTE STANDARD ERROR -- KNOWN RANKS CASE
# ------------------------------------------------------------------------------------
def compute_standard_error_known_ranks(
    y: np.ndarray, u: np.ndarray, se_method: str = "xavier"
) -> "tuple[np.ndarray, np.ndarray]":
    """
    compute_standard_error_known_ranks :
        computes the estimator (1), i.e. average of quantiles of the outcome for each rank

    WARNING :
        This function is a particular case of the previous one,
        it can probably be included with the right options, but I haven't took the time to do it.

    Args:
        y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
        u: np.array of the ranks
    """
    theta_hat, counterfactual_y = compute_theta_hat(u_hat=u, y=y)

    if se_method == "kernel":
        inv_density = 1.0 / kernel_density_estimator(x=np.quantile(y, u), data=y)
    elif se_method == "lewbel-schennach":
        u, inv_density = inv_density_ls(u_hat=u, y=y)
    elif se_method == "xavier":
        inv_density = inv_density_xavier(u_hat=u, y=y)
    else:
        raise ValueError(
            "'se_method' argument should be 'kernel', 'lewbel-schennach' or 'xavier'."
        )

    zeta = compute_zeta(u_hat=u, inv_density=inv_density, size=len(y))
    epsilon = theta_hat - counterfactual_y
    standard_error = np.sqrt(np.mean(zeta**2) + np.mean(epsilon**2)) / np.sqrt(len(y))

    return theta_hat, standard_error


# ------------------------------------------------------------------------------------
# BOOTSTRAP PROCEDURE
# ------------------------------------------------------------------------------------
def compute_bootstrap_sample(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    B: int = 1_000,
    cdf_method: str = "smoothed",
) -> np.ndarray:
    """
    compute_bootstrap_sample :
        computes the bootstrapped sample

    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome of untreated group at date 1.
        B (int): number of bootstrap samples
        cdf_method (str): can be "smoothed" or "standard" depending on the type of method for computation of the CDF

    Return:
        (np.array)
    """
    if cdf_method not in ["smoothed", "standard"]:
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
        u_hat = compute_counterfactual_ranks(
            x=x_sample, data=np.unique(z_sample), method=cdf_method
        )
        theta_hat, _ = compute_theta_hat(u_hat=u_hat, y=y_sample)
        results[b] = theta_hat
    return results


def compute_bootstrap_quantiles(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    B: int = 1_000,
    cdf_method: str = "smoothed",
    quantiles: list = [0.025, 0.975],
) -> "tuple[np.ndarray, np.ndarray]":
    """
    compute_bootstrap_quantiles :
        computes the bootstrapped quantiles of the estimator

    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome of untreated group at date 1.
        B (int): number of bootstrap samples
        method (str): can be "smoothed" or "standard" depending on the type of method for computation of the CDF
        quantiles (list): list of quantiles to compute

    Return:
        (np.array, np.array)
    """
    results = compute_bootstrap_sample(y=y, x=x, z=z, B=B, cdf_method=cdf_method)
    return np.quantile(results, quantiles)
