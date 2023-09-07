#!/bin/bash
module load singularity
export SINGULARITY_CACHEDIR=/scratch/jr22/pu6813/singularity
singularity pull --dir /scratch/jr22/pu6813/ docker://pabloabur/app
module unload singularity
