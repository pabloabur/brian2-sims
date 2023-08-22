#!/bin/bash
 
#PBS -P jr22
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=256GB
#PBS -l jobfs=10GB
#PBS -l walltime=04:00:00
#PBS -l wd

module load singularity
singularity run --nv /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/bal_stdp_cudadgx/ --code_path /scratch/jr22/pu6813/cudacodedgx/ --backend cuda_standalone balance_stdp > $PBS_JOBID.log
echo "Not so long simulation with high weight cap and less inhibition. Weights eventually become static" >> /scratch/jr22/pu6813/bal_stdp_cudadgx4/description.txt
