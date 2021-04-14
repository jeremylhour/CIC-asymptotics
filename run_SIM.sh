# --------------------------------
# MAIN SCRIPT TO RUN SIMULATIONS
# --------------------------------

EXPERIMENT=EXPONENTIAL # [EXPONENTIAL, GAUSSIAN]


########## DO NO MODIFY BELOW ##########

experiment=$(echo "$EXPERIMENT" | tr '[:upper:]' '[:lower:]')

echo CLEANING LEFTOVER FILES
rm -r input_configs_${EXPERIMENT}/
rm -f files_list_${EXPERIMENT}.txt
rm -r output/

echo CREATING CONFIG FILES
python3 DGP_${experiment}/${EXPERIMENT}_create_yml_files.py

echo RUNNING SIMULATIONS
mkdir output/
mkdir output/raw/
parallel -a files_list_${EXPERIMENT}.txt python3 DGP_$experiment/${EXPERIMENT}_simulation.py

echo CREATING RESULT TABLE
python3 DGP_${experiment}/${EXPERIMENT}_generate_latex.py

echo ZIPPING OUTPUT
zip -r output_${EXPERIMENT}.zip output/ 