sudo apt-get update
sudo apt-get install parallel

git clone https://github.com/jlhourENSAE/CIC-asymptotics.git

# normal run
cd CIC-asymptotics/
python3 simulation.py config_simulation.yml

# parallel run
cd CIC-asymptotics/
python3 create_yml_files.py # to create yml files
parallel -a files_list.txt python3 simulation.py
python3 generate_latex.py

# zipping output
sudo apt-get install parallel

zip -r zip -r output.zip output/ 