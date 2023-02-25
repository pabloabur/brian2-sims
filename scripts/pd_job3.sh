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

 #Protocol 2 with thalamic input
folder_name=/scratch/jr22/pu6813/pd_thal/
python3.10 run_simulation.py --backend cpp_standalone --save_path $folder_name --code_path /scratch/jr22/pu6813/code_thl/ PD --protocol 2 --w_in 7.5 --bg_freq 30 > $PBS_JOBID.log
echo "Potjans and Diesmann network under thalamic input. Repetitions of thalamic input is 10 and inhibitory weights is ${w_in} under ${bg_freq} background rate" >> ${folder_name}description.txt

conda deactivate
