"""
All simulations tools
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dgps import ExponentialDGP, GaussianDGP
from .estimators import estimator_unknown_ranks

np.random.seed(999)

# ------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------


def create_dgp_from_config(config):
    """
    create_dgp_from_config:
        Create a DGP instance based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing DGP parameters. Must contain keys:
            - "dgp": Type of DGP to create ("exponential" or "gaussian").
            - "n": Sample size for the generated data.
            - "params": Dictionary of parameters specific to the DGP type.

    Example:
        config = {
            "dgp": "exponential",
            "n": 1000,
            "params": {
                "lambda_x": 0.8,
                "lambda_z": 1,
                "alpha_y": 10
            }
        }

    Returns:
        ExponentialDGP or GaussianDGP: An instance of the specified DGP class.
    """

    if config["dgp"] == "exponential":
        return ExponentialDGP(n=config["n"], **config["params"])
    elif config["dgp"] == "gaussian":
        return GaussianDGP(n=config["n"], **config["params"])
    else:
        raise ValueError("Unknown DGP type specified in the configuration.")


def run_simulation(dgp, estimators, B: int = 1_000):
    """
    run_simulation:
        Run a simulation for a given DGP and a set of estimators.

    Args:
        dgp (DGP instance): The data generating process instance.
        estimators (dict): Dictionary of estimators to use.
        B (int): Number of simulations to run.

    Returns:
        list: A list of results from the simulations.
    """
    results = []
    bootstrap_quantiles = np.empty(shape=(B, 2))
    for b in tqdm(range(B)):
        y, z, x = dgp.generate()

        # for each estimator, compute the estimate and standard error
        res = {}
        for k, args in estimators.items():
            theta, sigma = estimator_unknown_ranks(y, x, z, **args)
            res[k] = {"theta": theta, "sigma": sigma}

        results.append(pd.DataFrame(res).unstack())

        _, bootstrap_quantiles[b] = estimator_unknown_ranks(
            y, x, z, method="smoothed", bootstrap_quantile=[0.025, 0.975]
        )

    # Compile the results
    results = (
        pd.concat(results, axis=1).T.replace({np.inf: np.nan}).dropna(axis=0, how="any")
    )
    return results, bootstrap_quantiles
