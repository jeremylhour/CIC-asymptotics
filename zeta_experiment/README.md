# zeta experiment

Scripts to test the asymptotic behavior of the estimator of zeta (eta in the paper).

```
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
parallel --j 5 -a job_list.txt python3 zeta_experiment/zeta_simulations.py

echo CREATING RESULT TABLE
python3 zeta_experiment/zeta_generate_latex.py

echo ZIPPING OUTPUT
zip -r output_zeta_$(date +'%d-%m-%Y').zip output/ 
```