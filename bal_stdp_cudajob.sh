#!/bin/bash
 
#PBS -P jr22
#PBS -q gpuvolta
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=64GB
#PBS -l jobfs=40GB
#PBS -l walltime=04:30:00
#PBS -l wd

#module unload intel-compiler intel-mkl python python2 python3 hdf5
##module load python3/3.10.4
##export PYTHONPATH=/g/data/your/directory/name/lib/python3.9/site-packages:$PYTHONPATH
#module load gcc/10.3.0
#module load cuda/12.0.0
#eval "$(conda shell.bash hook)"
#export CUDA_PATH=/apps/cuda/12.0.0
#export LD_LIBRARY_PATH="/apps/cuda/12.0.0/lib64:$LD_LIBRARY_PATH"
##export PATH="/apps/gcc/12.2.0/libexec/gcc/x86_64-pc-linux-gnu/12.2.0/:$PATH"
#conda activate b2s
#folder_name=/scratch/jr22/pu6813/bal_stdp_cuda/
##python run_simulation.py --backend cpp_standalone --save_path "${folder_name}" STDP --protocol 1
#python run_simulation.py --backend cuda_standalone --save_path "${folder_name}" --code_path /scratch/jr22/pu6813/code/ balance_stdp
###python simulations/maoi.py
##echo "Balanced network with STDP." >> ${folder_name}description.txt
#conda deactivate



module load singularity
singularity run --nv /scratch/jr22/pu6813/app_latest.sif --save_path /scratch/jr22/pu6813/bal_stdp_cuda/ --code_path /scratch/jr22/pu6813/cudacode/ --backend cuda_standalone balance_stdp > $PBS_JOBID.log
