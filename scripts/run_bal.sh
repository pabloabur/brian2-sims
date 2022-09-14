#!/bin/bash
folder_name="sim_data/balance_exp"
mkdir -p $folder_name && echo "Balanced network with minifloat." >> ${folder_name}/description.txt
for wi in {70..115}; do
    for trial in {0..9}; do
        sim_name=$(date +"${folder_name}/%d-%m_%Hh%Mm%Ss")
        python simulations/SLIF_balanced_tutorial.py --wi $wi --trial $trial --path ${sim_name}/ --quiet
    done
done
