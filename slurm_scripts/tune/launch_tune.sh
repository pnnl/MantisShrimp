#!/bin/bash

#SBATCH -A mantis_shrimp
#SBATCH -J MS_tune    # create a short name for your job
#SBATCH -p dl
#SBATCH --nodes=1 #number of nodes (of machines)
#SBATCH --exclusive
#SBATCH -t 96:0:0 #h:m:s
#SBATCH -o /rcfs/projects/mantis_shrimp/mantis_shrimp/slurm_scripts/tune/o.txt
#SBATCH -e /rcfs/projects/mantis_shrimp/mantis_shrimp/slurm_scripts/tune/e.txt

echo "Environment Variables"
echo
printenv


# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc 
source ~/.bashrc

module purge
# unlimit

module load python/miniconda3.9
module load cuda/11.8
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate torch2.0

cd /rcfs/projects/mantis_shrimp/mantis_shrimp/NOTEBOOKS/

python tune_V1p0.py

echo "All Done!"
