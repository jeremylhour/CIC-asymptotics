import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle

from cic_asymptotics import (
    get_performance_report,
    create_dgp_from_config,
    run_simulation,
    DEFAULT_ESTIMATORS,
    DEFAULT_CONFIG,
    OUTPATH,
    DEFAULT_SIMULATION_SIZE,
)

if __name__ == "__main__":
    # Defines the number of simulations
    B = DEFAULT_SIMULATION_SIZE

    # Defines the DGP
    dgp = create_dgp_from_config(DEFAULT_CONFIG)

    # Estimators
    estimators = DEFAULT_ESTIMATORS

    results, bootstrap_quantiles = run_simulation(
        dgp=dgp,
        estimators=estimators,
        B=B,
    )

    report = get_performance_report(
        results.xs("theta", level=1, axis=1),
        theta0=dgp.theta0,
        n_obs=dgp.n,
        histogram=False,
        sigma=results.xs("sigma", level=1, axis=1),
        verbose=True,
        bootstrap_quantiles=bootstrap_quantiles,
    )

    filename = f"{dgp.name}_B={B}_n={dgp.n}.p"
    pickle.dump(report, open(OUTPATH / filename, "wb"))
