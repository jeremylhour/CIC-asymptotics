# CIC-asymptotics
Codes for Change-in-Change Asymptotics project with Xavier D'Haultfoeuille and Martin Mugnier.

## Main file:
[DGP_exponential/EXPONENTIAL_simulation.py](DGP_exponential/EXPONENTIAL_simulation.py) is the main file for running the simulations for the exponential DGP. It loads the configuration from the YAML file [DGP_exponential/EXPONENTIAL_example.yml](DGP_exponential/EXPONENTIAL_example.yml). It works in the same manner for other DGPs.

Can be ran from terminal by being in the main folder and entering:
```
python3 DGP_exponential/EXPONENTIAL_simulation.py DGP_exponential/EXPONENTIAL_example.yml
```

## Parallel computing:
Simulations can be ran in parallel from the terminal using GNU parallel tool. First, YAML files need to be created using e.g. the script [DGP_gaussian/GAUSSIAN_create_yml_files.py](DGP_gaussian/GAUSSIAN_create_yml_files.py). Then, simulations can be ran in parallel inputing the YAML file names from the automatically created list.

```
python3 DGP_gaussian/GAUSSIAN_create_yml_files.py # to create yml files
parallel -a files_list_GAUSSIAN.txt python3 DGP_gaussian/GAUSSIAN_simulation.py # to run
```

Different outputs are produced from this scripts, among which the summary of the results in a pickle file. Then to aggregate them all and create a latex table, run:

```
python3 DGP_gaussian/GAUSSIAN_generate_latex.py # to create latex table
```

If 'parallel' is not installed, run before:

```
sudo apt-get update
sudo apt-get install parallel
```

## Important folders:
- [functions](functions/): contains all the necessary functions splitted by theme,
- [test_scripts](test_scripts/): contains scripts that were used to develop the functions.

## Other folders:
- input_configs: will be created when running a create_yml_files.py file (for example [DGP_gaussian/GAUSSIAN_create_yml_files.py](DGP_gaussian/GAUSSIAN_create_yml_files.py)) and will contain the necessary YAML files,
- output: will also be created when running [simulation.py](simulation.py). It will contain simulations results (log files, histograms, latex table) as well as saved raw results as pickle format (inside the "raw" folder).
Both folders should not be pushed to git repo.
