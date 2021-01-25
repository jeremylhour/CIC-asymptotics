sudo apt-get update
sudo apt-get install parallel
sudo apt-get install zip

git clone https://github.com/jlhourENSAE/CIC-asymptotics.git

# normal run
cd CIC-asymptotics/
python3 simulation.py config_simulation.yml

# parallel run -- Exponential
cd CIC-asymptotics/
python3 DGP_exponential/EXPONENTIAL_create_yml_files.py # to create yml files
parallel -a files_list.txt python3 DGP_exponential/EXPONENTIAL_simulation.py
python3 DGP_exponential/EXPONENTIAL_generate_latex.py


# parallel run -- Gaussian
cd CIC-asymptotics/
python3 DGP_gaussian/GAUSSIAN_create_yml_files.py # to create yml files
parallel -a files_list_GAUSSIAN.txt python3 DGP_gaussian/GAUSSIAN_simulation.py # to run
python3 DGP_gaussian/GAUSSIAN_generate_latex.py # to create latex table

# zipping output
zip -r output.zip output/ 