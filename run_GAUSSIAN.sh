cd Documents/code/CIC-asymptotics/

rm -rf /input_configs_GAUSSIAN
rm -f files_list_GAUSSIAN.txt
rm -rf /output

echo 'Running simulations for Gaussian DGP \n'

echo 'Creating YAML config files \n'
python3 DGP_gaussian/GAUSSIAN_create_yml_files.py # to create yml files

echo 'Running simulations ...' 
parallel -a files_list_GAUSSIAN.txt python3 DGP_gaussian/GAUSSIAN_simulation.py # to run

echo 'Creating LaTeX table.' 
python3 DGP_gaussian/GAUSSIAN_generate_latex.py # to create latex table

# zipping output
zip -r output.zip output/ 