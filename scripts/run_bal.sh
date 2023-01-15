#!/bin/bash
# code below maintained for reference
#folder_name="sim_data/balance_exp"
#mkdir -p $folder_name && echo "Balanced network with minifloat." >> ${folder_name}/description.txt
#for wi in {70..115}; do
#    for trial in {0..9}; do
#        sim_name=$(date +"${folder_name}/%d-%m_%Hh%Mm%Ss")
#        python simulations/SLIF_balanced_tutorial.py --wi $wi --trial $trial --path ${sim_name}/ --quiet
#    done
#done
folder_name="sim_data/balance_exp"
for wi in $(seq -f "%.6f" .03125 0.03125 1); do
    for trial in {0..4}; do
        sim_name=$(date +"${folder_name}/%d-%m_%Hh%Mm%Ss")
        python run_simulation.py --quiet --save_path $sim_name --backend cpp_standalone balance --w_perc $wi
    done
done
echo "Balanced network with models." >> ${folder_name}/description.txt
