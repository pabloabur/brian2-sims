#!/bin/bash
 
#PBS -P jr22
#PBS -q normal
#PBS -l ncpus=8
#PBS -l mem=64GB
#PBS -l jobfs=8GB
#PBS -l walltime=08:00:00
#PBS -l wd
 
# Load module, always specify version number.
module load gcc/11.1.0
eval "$(conda shell.bash hook)"
conda activate neu_nets

# Protocol 1 to find AI landscape
folder_name=/scratch/jr22/pu6813/pd_async_irreg_fp8/
w_ins=(16.25 17.50)
for w_in in "${w_ins[@]}"; do
   for bg_freq in {20..80..10}; do
        python3.10 run_simulation.py --backend cpp_standalone --save_path "${folder_name}win${w_in}_bg${bg_freq}/" --code_path /scratch/jr22/pu6813/code/ PD --protocol 1 --w_in $w_in --bg_freq $bg_freq > $PBS_JOBID.log
   done
done
echo "Potjans and Diesmann network and AI landscape. Different folder were create for pair of parameters" >> ${folder_name}description.txt

conda deactivate
