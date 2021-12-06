# --------------------------------------------------------------
# CIC ASYMPTOTICS
#
# MAIN SCRIPT TO RUN SIMULATIONS
#
# @author: jeremy.l.hour@ensae.fr
# --------------------------------------------------------------

EXPERIMENT=GAUSSIAN # [EXPONENTIAL, GAUSSIAN]
INSTALL=false
ZETA=true


########## DO NO MODIFY BELOW ##########
if [ "$INSTALL" = true ] ; then
    echo INSTALLING PACKAGES
    sudo apt-get update
    sudo apt-get install parallel -y
    sudo apt-get install zip -y
fi

experiment=$(echo "$EXPERIMENT" | tr '[:upper:]' '[:lower:]')

echo CLEANING LEFTOVER FILES
rm -r input_configs_${EXPERIMENT}/
rm -f job_list.txt
rm -r output/

echo CREATING CONFIG FILES FOR INPUT
python3 DGP_${experiment}/${EXPERIMENT}_create_yml_files.py

echo RUNNING SIMULATIONS
mkdir output/
mkdir output/raw/
parallel --j 5 -a job_list.txt python3 DGP_$experiment/${EXPERIMENT}_simulation.py

echo CREATING RESULT TABLE
python3 DGP_${experiment}/${EXPERIMENT}_generate_latex.py

echo ZIPPING OUTPUT
zip -r output_${EXPERIMENT}_$(date +'%d-%m-%Y').zip output/ 


########## FOR ZETA ESTIMATOR ##########
if [ "$ZETA" = true ] ; then
    EXPERIMENT=EXPONENTIAL
    experiment=$(echo "$EXPERIMENT" | tr '[:upper:]' '[:lower:]')
    
    echo ZETA EXPERIMENT
    rm -r input_configs_${EXPERIMENT}/
    rm -f job_list.txt
    rm -r output_zeta/
    
    echo CREATING CONFIG FILES FOR INPUT
    python3 DGP_${experiment}/${EXPERIMENT}_create_yml_files.py
    
    echo RUNNING SIMULATIONS
    mkdir output_zeta/
    mkdir output_zeta/raw/
    parallel --j 5 -a job_list.txt python3 zeta_experiment/simulations.py
    
    echo CREATING RESULT TABLE
    python3 zeta_experiment/zeta_generate_latex.py
    
    echo ZIPPING OUTPUT
    zip -r output_zeta_$(date +'%d-%m-%Y').zip output/ 
fi