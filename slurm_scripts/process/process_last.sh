#!/bin/sh
#SBATCH -A mantis_shrimp    #account name-- this might be a WP
#SBATCH -t 96:0:0    #time h:m:s
#SBATCH -N 1    #Number of nodes
#SBATCH -n 4    #Number of cores
#SBATCH -J process    #name of the job
#SBATCH -o "./%j.o"    #name of the output file
#SBATCH -e "./%j.e"    #name of the error file
#SBATCH -p shared

cd ~

module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate torch2.0.0
cd /rcfs/projects/mantis_shrimp/mantis_shrimp/

python process.py
