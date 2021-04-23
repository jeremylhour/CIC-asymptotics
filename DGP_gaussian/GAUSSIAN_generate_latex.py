#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate latex from the raw results output -- Gaussian edition.

Created on Sun Jan 24 18:34:39 2021

@author: jeremylhour
"""
import os
import yaml
import pickle


if __name__ == '__main__':
    ########## LOAD MAIN CONFIG ##########
    config_file = os.path.join(os.getcwd(), "DGP_gaussian/GAUSSIAN_config.yml")
    
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    
    B = config["nb_simu"]
    variance_x = config["variance_x"]
    mu_x = config["mu_x"]
    
    colours = False
    
    ########## CREATING TABLE ##########
    
    ########## PARAMETERS ##########
    models = [
        "standard_kernel",
        "standard_xavier",
        "smooth_kernel",
        "smooth_ls",
        "smooth_xavier",
    ]
    metrics_set = ["bias", "RMSE", "Coverage rate", "CI size"]
    
    color_dico = dict(
        {
            0.1: "LightCyan",
            0.2: "LightRed",
            0.3: "LightGreen",
            0.5: "LightYellow",
            0.7: "LightBlue",
            0.8: "LightRed",
            0.9: "LightGreen",
        }
    )  # dico for coloring the blocks
    digits = 3  # rounding
    
    ########## CORE CODE ##########
    counter = 0
    
    f = open("output/GAUSSIAN_simulation.tex", "w")
    
    for j in mu_x:
        for i in variance_x:
            pickle_file = (
                "output/raw/gaussian_simulations_B="
                + str(B)
                + "_mu_x="
                + str(j)
                + "_variance_x="
                + str(i)
                + ".p"
            )
            try:
                result = pickle.load(open(pickle_file, "rb"))
            except:
                continue
    
            param_line = (
                r" & \multicolumn{"
                + str(len(metrics_set) * len(result))
                + "}{c}{"
                + "$\mu_X$="
                + str(j)
                + ", $\sigma_X^2$="
                + str(i)
                + " -- $b_1=b_2$="
                + str(round(1 - 1/i, 2))
                + "}"
            )
            param_line = param_line + "\\\\"
            for model in models:
                counter += 1
                string = model.replace("_", " ")
                item = "model"
                sample_line = " "
                header = r"\begin{longtable}{l|"
                for sample_size in result:
                    sample_line = (
                        sample_line
                        + r" & \multicolumn{"
                        + str(len(metrics_set))
                        + "}{c}{"
                        + "n="
                        + str(sample_size)
                        + "}"
                    )
                    header = header + ("c" * len(metrics_set))
                    for metric in metrics_set:
                        string = (
                            string
                            + " & "
                            + str(round(result[sample_size][metric][model], digits))
                        )
                        item = item + " & " + metric
                string = string + "\\\\"
                item = item + "\\\\"
                sample_line = sample_line + "\\\\"
                header = header + "}"
    
                ### WRITING
                if counter == 1:
                    f.write(header)
                    f.write("\n")
                    f.write(r"\caption{}\\")
                    f.write("\n")
                    f.write(r"\toprule")
                    f.write("\n")
                    f.write(sample_line)
                    f.write("\n")
                    f.write(item)
                    f.write("\n")
                    f.write(r"\hline")
                    f.write("\n")
                if model == models[0]:
                    f.write(r"\hline")
                    f.write("\n")
                    f.write(param_line)
                    f.write("\n")
                if colours:
                    f.write(r"\rowcolor{" + color_dico[i] + "} ")
                f.write(string)
                f.write("\n")
    
    f.write(r"\bottomrule")
    f.write("\n")
    f.write(r"\end{longtable}")
    f.write("\n")
    f.close()