import json
import sys
import yaml
import logging
from pathlib import Path

from cic_asymptotics import (
    get_performance_report,
    create_dgp_from_config,
    run_simulation,
    DEFAULT_ESTIMATORS,
    DEFAULT_CONFIG,
    OUTPATH,
    DEFAULT_SIMULATION_SIZE,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    logging.info(f"Loaded config from: {cfg_path}")

    # Load the parameters from the config file
    B = cfg.get("simulation_size", DEFAULT_SIMULATION_SIZE)
    logging.info(f"Simulation size (B): {B}")
    dgp = create_dgp_from_config(cfg.get("config", DEFAULT_CONFIG))
    logging.info(f"DGP created: {dgp.name} with n = {dgp.n}")
    estimators = cfg.get("estimators", DEFAULT_ESTIMATORS)
    logging.info(f"Estimators loaded: {list(estimators.keys())}")
    output_path = Path(cfg.get("output_path", OUTPATH))
    logging.info(f"Output path set to: {output_path}")

    # Runs simulations
    logging.info("Running simulations...")
    results, bootstrap_quantiles = run_simulation(
        dgp=dgp,
        estimators=estimators,
        B=B,
    )

    # Process results and generate performance report
    report = get_performance_report(
        results.xs("theta", level=1, axis=1),
        theta0=dgp.theta0,
        n_obs=dgp.n,
        histogram=False,
        sigma=results.xs("sigma", level=1, axis=1),
        verbose=False,
        bootstrap_quantiles=bootstrap_quantiles,
    )
    logging.info("Performance report generated.")

    # Save the results to a json file
    filename = f"{dgp.name}_B={B}_n={dgp.n}"
    with open(output_path / f"{filename}.json", "w") as f:
        json.dump(report, f, indent=4)

    logging.info("Done.")
