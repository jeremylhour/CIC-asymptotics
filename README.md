# CIC-asymptotics
Codes for the 'Change-in-Change Asymptotics' project with Xavier D'Haultfoeuille and Martin Mugnier.

@author : jeremy.l.hour@ensae.fr

## Main file

[DGP_exponential/EXPONENTIAL_simulation.py](DGP_exponential/EXPONENTIAL_simulation.py) is the main file for running the simulations for the exponential DGP. It loads the configuration from the YAML file given as argument in the shell (could be [DGP_exponential/EXPONENTIAL_example.yml](DGP_exponential/EXPONENTIAL_example.yml) for example). It works in the same manner for other DGPs.

Here is how to run it. From within the main folder, enter :
```
python3 DGP_exponential/EXPONENTIAL_simulation.py DGP_exponential/EXPONENTIAL_example.yml
```

You can run an example by doing :
```
python3 src/simulations.py
```

## Parallel computing

Simulations can be ran in parallel from the terminal using GNU parallel tool. First, YAML files need to be created using e.g. the script [DGP_gaussian/GAUSSIAN_create_yml_files.py](DGP_gaussian/GAUSSIAN_create_yml_files.py). Then, simulations can be ran in parallel inputing the YAML file names from the automatically created list.
```
python3 DGP_gaussian/GAUSSIAN_create_yml_files.py # to create yml files
parallel -a files_list_GAUSSIAN.txt python3 DGP_gaussian/GAUSSIAN_simulation.py # to run
```

Different outputs are produced from this script, among which the summary of the results in a pickle file. Then to aggregate them all and create a latex table, run:
```
python3 DGP_gaussian/GAUSSIAN_generate_latex.py # to create latex table
```

The file [run_SIM.sh](run_SIM.sh) compiles these instructions and offers a way to reproduce the simulations from the paper. Assign either 'EXPONENTIAL' or 'GAUSSIAN' to the EXPERIMENT variable and run from the main folder (make sure it can be executed):
```
./run_SIM.sh
```

To fully run this script, 'parallel' and 'zip' need to be installed first by running :
```
sudo apt-get update
sudo apt-get install parallel
sudo apt-get install zip
```

## Important folders

- [src](src/): contains all the necessary functions broken down by theme,
- [test_scripts](test_scripts/): contains scripts that were used to develop the functions.

## Other folders

- input_configs: will be created when running a create_yml_files.py file (for example [DGP_gaussian/GAUSSIAN_create_yml_files.py](DGP_gaussian/GAUSSIAN_create_yml_files.py)) and will contain the necessary YAML files,
- output: will also be created when running [simulation.py](simulation.py). It will contain simulations results (log files, histograms, latex table) as well as saved raw results as pickle format (inside the "raw" folder).
Both folders should not be pushed to git repo.

## License

The files and documents available in this repository are under the Etalab Open License version 2.0 (see the
[license](./LICENSE)).

The use of these files is free of charge and without restriction, at the sole condition of citing it as :
```
D'Haultfoeuille, X., L'Hour, J. and Mugnier, M.  - https://github.com/jlhourENSAE/CIC-asymptotics, 2021
```
