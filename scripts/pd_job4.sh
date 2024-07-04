#!/bin/bash
 
#PBS -P jr22
#PBS -q normal
#PBS -l ncpus=8
#PBS -l mem=64GB
#PBS -l jobfs=8GB
#PBS -l walltime=08:00:00
#PBS -l wd

module load singularity
#singularity run /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/pd_thal1/ --code_path /scratch/jr22/pu6813/code1/ --backend cpp_standalone PD --protocol 1 --w_in 7.5 --bg_freq 30 > $PBS_JOBID.log
singularity run /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/pd_thal2/ --code_path /scratch/jr22/pu6813/code/ --backend cpp_standalone PD --protocol 2 --w_in 7.5 --bg_freq 30 > $PBS_JOBID.log
