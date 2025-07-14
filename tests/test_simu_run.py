import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from scipy.stats import expon, pareto

from cic_asymptotics import (
    estimator_unknown_ranks,
    generate_data,
    analytical_theta,
    get_performance_report,
)

if __name__ == "__main__":
    print("THIS IS AN EXAMPLE USING EXPONENTIAL DGP")

    B = 100
    lambda_x = 0.8
    lambda_z = 1
    alpha_y = 10
    PRINT = False

    if PRINT:
        print(
            f"lambda_x={lambda_x} -- lambda_z={lambda_z} -- alpha_y={alpha_y}",
            f"Parameter values give b_2={round(1 - lambda_x / lambda_z, 2)}",
            f"Parameter values give d_2={round(1 / alpha_y, 2)}",
            f"So b_2+d_2={round(1 - lambda_x / lambda_z + 1 / alpha_y, 2)}",
            "--- Remember, b_2 + d_2 should be below .5 for Theorem 2 to apply",
            "--- and below 1 for theta_0 to be finite.",
            sep="\n",
        )

    nb_estimators = 5
    sample_size_set = [500]
    big_results = {}

    for sample_size in sample_size_set:
        print("Running {} simulations with sample size {}...".format(B, sample_size))

        np.random.seed(999)
        results, sigma = (
            np.empty(shape=(B, nb_estimators)),
            np.empty(shape=(B, nb_estimators)),
        )

        start_time = time.time()
        for b in tqdm(range(B)):
            y, z, x = generate_data(
                distrib_y=pareto(b=alpha_y, loc=-1),
                distrib_z=expon(scale=1 / lambda_z),
                distrib_x=expon(scale=1 / lambda_x),
                size=sample_size,
            )
            # Estimator and standard error
            theta_standard, sigma_standard = estimator_unknown_ranks(
                y, x, z, method="standard"
            )
            theta_standard_x, sigma_standard_x = estimator_unknown_ranks(
                y, x, z, method="standard", se_method="xavier"
            )

            theta_smooth, sigma_smooth = estimator_unknown_ranks(
                y, x, z, method="smoothed", se_method="kernel"
            )
            theta_ls, sigma_ls = estimator_unknown_ranks(
                y, x, z, method="smoothed", se_method="lewbel-schennach"
            )
            theta_x, sigma_x = estimator_unknown_ranks(
                y, x, z, method="smoothed", se_method="xavier"
            )

            # Collecting results
            results[b,] = [
                theta_standard,
                theta_standard_x,
                theta_smooth,
                theta_ls,
                theta_x,
            ]
            sigma[b,] = [
                sigma_standard,
                sigma_standard_x,
                sigma_smooth,
                sigma_ls,
                sigma_x,
            ]

            # Checking division error
            if np.isinf(sigma_smooth) or np.isnan(sigma_ls):
                print(" -- error for this iteration")
                results[b,] = [np.nan] * nb_estimators
                sigma[b,] = [np.nan] * nb_estimators
        print(
            f"Temps d'exécution total : {(time.time() - start_time):.2f} secondes ---"
        )

        ########## POST-PROCESSS ##########
        results = pd.DataFrame(results)
        results.dropna(axis=0, inplace=True)

        sigma = pd.DataFrame(sigma)
        sigma.dropna(axis=0, inplace=True)

        theta0 = analytical_theta(alpha_y=alpha_y, lambda_z=lambda_z, lambda_x=lambda_x)

        y_hat = pd.DataFrame(
            {
                "Standard kernel": results[0],
                "Standard Xavier": results[1],
                "Smooth kernel": results[2],
                "Smooth LS": results[3],
                "Smooth Xavier": results[4],
            }
        )

        sigma_df = pd.DataFrame(
            {
                "Standard kernel": sigma[0],
                "Standard Xavier": sigma[1],
                "Smooth Kernel": sigma[2],
                "Smooth LS": sigma[3],
                "Smooth Xavier": sigma[4],
            }
        )

        big_results[sample_size] = get_performance_report(
            y_hat,
            theta0,
            n_obs=sample_size,
            histogram=False,
            sigma=sigma_df,
            print_report=PRINT,
        )

        if PRINT:
            print(big_results)
