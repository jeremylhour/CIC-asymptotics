# normal run
cd Documents/code/CIC-Asymptotics/
python3 simulation.py config_simulation.yml

# parallel run
cd Documents/code/CIC-Asymptotics/
python3 create_yml_files.py # to create yml files
parallel -a files_list.txt python3 simulation.py
python3 generate_latex.py