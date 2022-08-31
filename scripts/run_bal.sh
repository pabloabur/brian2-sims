#!/bin/bash
folder_name="sim_data/balance_exp"
for wi in {20..80..4}; do
    path_name=$(date +"$folder_name/%d-%m_%Hh%Mm%Ss")
    mkdir -p $path_name
    echo "Balanced network. Each folder contains" >> $path_name/description.txt
    echo "trials (same net, different seed) for each" >> $path_name/description.txt
    echo "inhibitory weight. Model is minifloat" >> $path_name/description.txt
    python orca_workspace/SLIF_balanced_tutorial.py $wi --path $path_name --quiet
done
