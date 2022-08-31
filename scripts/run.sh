#!/bin/bash
folder_name="sim_data/gap_exp2"
echo "Creating $folder_name..."
mkdir -p $folder_name && echo "Experiment with time gaps over many trials, now with more points and new readout." >> ${folder_name}/description.txt
#echo "w_factor was not increased on testing phase." >> ${folder_name}/description.txt
for gap in {0..40..3}; do
    for trial in {0..9}; do
        sim_name=$(date +"${folder_name}/%d-%m_%Hh%Mm%Ss")
        python orca_workspace/SLIF_extrapolation.py --gap $gap --trial $trial --path ${sim_name}/ --quiet
    done
done
