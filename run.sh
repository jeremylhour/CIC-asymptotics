# normal run
cd Documents/code/CIC-Asymptotics/
python3 simulation.py config_simulation.yml

# parallel run
cd Documents/code/CIC-Asymptotics/
Python3 create_uml_files.py # to create yml files
parallel -a files_list.txt python3 simulation.py