cd Documents/code/CIC-asymptotics/

rm -rf /input_configs_GAUSSIAN
rm -f files_list_GAUSSIAN.txt
rm -rf /output

python3 DGP_gaussian/GAUSSIAN_create_yml_files.py # to create yml files
parallel -a files_list_GAUSSIAN.txt python3 DGP_gaussian/GAUSSIAN_simulation.py # to run
python3 DGP_gaussian/GAUSSIAN_generate_latex.py # to create latex table

# zipping output
zip -r output.zip output/ 