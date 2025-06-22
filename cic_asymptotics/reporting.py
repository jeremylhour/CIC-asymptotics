import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


# ------------------------------------------------------------------------------------
# PERFORMANCE REPORT
# ------------------------------------------------------------------------------------
def compute_metrics(
    y_hat,
    theta0: float,
    n_obs: int,
    sigma=None,
    bootstrap_quantiles=None,
):
    """
    compute_metrics:
        computes the metrics for simulations.

    Args:
        y_hat (np.array): B x K np.array of B simulations for K estimators
        theta0 (float): scalar, true value of theta
        n_obs (int): sample size used during simulations
        sigma (pd.DataFrame): B x K np.array of B simulations for K standard errors
        bootstrap_quantiles (np.array): lower bound and upper bound of confidence interval computed by bootstrap
    """
    if sigma is None:
        sigma = np.ones(y_hat.shape)

    y_centered = y_hat - theta0
    metrics = {
        "theta0": theta0,
        "n_simu": len(y_hat),
        "n_obs": n_obs,
        "bias": y_centered.mean(axis=0).to_dict(),
        "MAE": abs(y_centered).mean(axis=0).to_dict(),
        "RMSE": y_centered.std(axis=0).to_dict(),
        "Coverage rate": (abs(y_centered / sigma) < norm.ppf(0.975))
        .mean(axis=0)
        .to_dict(),
        "Quantile .95": (np.sqrt(n_obs) * y_centered)
        .quantile(q=0.95, axis=0)
        .to_dict(),
        "CI size": (2 * norm.ppf(0.975) * sigma.mean(axis=0)).to_dict(),
    }

    if bootstrap_quantiles is not None:
        metrics["Coverage rate"]["bootstrap"] = (
            (bootstrap_quantiles[:, 0] < theta0) & (bootstrap_quantiles[:, 1] > theta0)
        ).mean()
        metrics["CI size"]["bootstrap"] = (
            bootstrap_quantiles[:, 1] - bootstrap_quantiles[:, 0]
        ).mean()

    return metrics


def print_report(report):
    """
    print_report:
        prints the report for simulations,
        computes bias, MSE, MAE and coverage rate.

    Args:
        report (dict): dictionary containing the metrics
    """
    print("Theta_0 : {:.2f}".format(report.get("theta0")))
    print("Number of simulations : {} \n".format(report.get("n_simu")))
    print("Sample size : {} \n".format(report.get("n_obs")))
    for metric in ["bias", "MAE", "RMSE", "Coverage rate", "CI size", "Quantile .95"]:
        print(metric + ": ")
        for model in report.get(metric):
            print("- {} : {:.3f}".format(model, report.get(metric).get(model, np.nan)))
        print("\n")


def get_performance_report(
    y_hat,
    theta0: float,
    n_obs: int,
    sigma=None,
    bootstrap_quantiles=None,
    file: str = "default_output_file",
    histogram: bool = True,
    verbose: bool = True,
):
    """
    performance_report:
        creates the report for simulations,
        computes bias, MSE, MAE and coverage rate.

    Args:
        y_hat (np.array): B x K np.array of B simulations for K estimators
        theta0 (float): scalar, true value of theta
        n_obs (int): sample size used during simulations
        histogram (bool): whether to draw the histograms
        sigma (pd.DataFrame): B x K np.array of B simulations for K standard errors
        file (str): where to save the report
        bootstrap_quantiles (np.array): lower bound and upper bound of confidence interval computed by bootstrap
    """
    report = compute_metrics(
        y_hat=y_hat,
        theta0=theta0,
        n_obs=n_obs,
        sigma=sigma,
        bootstrap_quantiles=bootstrap_quantiles,
    )

    if verbose:
        print_report(report)

    with open(file + ".txt", "a") as f:
        f.write("\n")
        f.write("Theta_0: {:.2f} \n".format(report["theta0"]))
        for metric in [
            "bias",
            "MAE",
            "RMSE",
            "Coverage rate",
            "CI size",
            "Quantile .95",
        ]:
            f.write(metric + ": \n")
            for model in report.get(metric):
                f.write(
                    "- {}: {:.4f} \n".format(
                        model, report.get(metric).get(model, np.nan)
                    )
                )
            f.write("\n")

    if histogram:
        if sigma is None:
            sigma = np.ones(y_hat.shape)

        y_centered = y_hat - theta0
        num_bins = 50
        for model in y_centered.columns:
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(
                np.sqrt(n_obs) * y_centered[model], num_bins, density=1
            )
            norm_fit = norm.pdf(bins, scale=np.sqrt(n_obs) * sigma[model].mean())
            ax.plot(bins, norm_fit, "--")
            ax.set_xlabel(r"$n^{1/2}$ ($\hat \theta$ - $\theta_0$)")
            ax.set_ylabel("Probability density")
            ax.set_title(r"Histogram for model: " + model)
            fig.tight_layout()
            plt.savefig(file + "_n=" + str(n_obs) + "_" + model + ".jpg", dpi=(96))
    return report


def save_latex_table(
    results,
    file: str,
    models=["standard", "smoothed", "smoothed_lewbel-schennach"],
    digits: int = 3,
):
    """
    latex_table:
        outputs a latex table from a list of results

    Args:
        results: list of results based on the format results[sample_size][metric][model]
        file (str): name of the output file
        models (list of str): list of the models
        digits (int): defines the precision
    """
    metrics_set = ["bias", "MAE", "RMSE", "Coverage rate", "Quantile .95"]
    k = 0

    with open(file + ".tex", "a") as f:
        f.write("\n")
        f.write(r"\begin{table}")
        f.write("\n")

        for model in models:
            k += 1
            string = model
            item = "model"
            sample_line = " "
            header = r"\begin{tabular}{l|"
            for sample_size in results:
                sample_line = (
                    sample_line
                    + r" & \multicolumn{"
                    + str(len(metrics_set))
                    + "}{c}{"
                    + str(sample_size)
                    + "}"
                )
                header = header + ("c" * len(metrics_set))
                for metric in metrics_set:
                    string = (
                        string
                        + " & "
                        + str(round(results[sample_size][metric][model], digits))
                    )
                    item = item + " & " + metric
            string = string + "\\\\"
            item = item + "\\\\"
            sample_line = sample_line + "\\\\"
            header = header + "}"
            ### WRITING
            if k == 1:
                f.write(header)
                f.write("\n")
                f.write(r"\toprule")
                f.write("\n")
                f.write(sample_line)
                f.write("\n")
                f.write(item)
                f.write("\n")
                f.write(r"\hline")
                f.write("\n")
            f.write(string)
            f.write("\n")

        f.write(r"\bottomrule")
        f.write("\n")
        f.write(r"\end{tabular}")
        f.write("\n")
        f.write(r"\end{table}")
        f.write("\n")
    return None
