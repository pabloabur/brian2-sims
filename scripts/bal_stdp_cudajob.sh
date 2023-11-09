#!/bin/bash
 
#PBS -P jr22
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=256GB
#PBS -l jobfs=10GB
#PBS -l walltime=05:00:00
#PBS -l wd

module load singularity
singularity run --nv /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/bal_stdp_cudadgx/ --code_path /scratch/jr22/pu6813/cudacodedgx/ --backend cuda_standalone balance_stdp --protocol 1 --w_max .5 --event_condition 'abs(Ca) > 0.02' --we 0.25 > $PBS_JOBID.log
#singularity run --nv /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/bal_stdp_cudadgx/ --code_path /scratch/jr22/pu6813/cudacodedgx/ --backend cuda_standalone balance_stdp --protocol 1 --w_max .5 --event_condition 'abs(Ca) > 0.02' --we 0.25 --ca_decays '(15*rand() + 10)*ms' > $PBS_JOBID.log
#singularity run --nv /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/bal_stdp_cudadgx_mad/ --code_path /scratch/jr22/pu6813/cudacodedgx_mad/ --backend cuda_standalone balance_stdp --protocol 2 --event_condition 'abs(Ca) > 0.02' > $PBS_JOBID.log
#singularity run --nv /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/bal_stdp_cudadgx_mad/ --code_path /scratch/jr22/pu6813/cudacodedgx_mad/ --backend cuda_standalone balance_stdp --protocol 2 --event_condition 'abs(Ca) > 0.02' --alpha 0.1420 --tsim 13 > $PBS_JOBID.log
