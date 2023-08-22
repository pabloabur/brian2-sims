#!/bin/bash
 
#PBS -P jr22
#PBS -q normal
#PBS -l ncpus=12
#PBS -l mem=128GB
#PBS -l jobfs=40GB
#PBS -l walltime=04:30:00
#PBS -l wd

module load singularity
singularity run /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/bal_stdp/ --code_path /scratch/jr22/pu6813/code/ --backend cpp_standalone balance_stdp > $PBS_JOBID.log
