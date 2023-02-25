#!/bin/bash
 
#PBS -P jr22
#PBS -q normal
#PBS -l ncpus=8
#PBS -l mem=64GB
#PBS -l jobfs=8GB
#PBS -l walltime=48:00:00
#PBS -l wd
 
# Load module, always specify version number.
module load gcc/11.1.0
eval "$(conda shell.bash hook)"
conda activate neu_nets
 
# Run Python applications
folder_name=/scratch/jr22/pu6813/lsm/
for size in $(seq 128 256 4992); do
    for trial in {1..10}; do
        sim_name=$(date +"$folder_name/%d-%m_%Hh%Mm%Ss")
        python3.10 run_simulation.py --save_path $sim_name --code_path /scratch/jr22/pu6813/code/ --backend cpp_standalone --quiet LSM --precision fp8 --size $size --trial $trial
    done
done
conda deactivate
