#!/bin/bash

#SBATCH -A mantis_shrimp
#SBATCH -J MS_train    # create a short name for your job
#SBATCH -p dl
#SBATCH --nodes=8 #number of nodes (of machines)
###SBATCH --ntasks=16 #total number of tasks across all machines 
###SBATCH --cpus-per-task=8  # total number of CPU cores allocated per task
###SBATCH --gres=gpu:1            # number of gpus per task/process 
#SBATCH --exclusive
#SBATCH -t 96:0:0 #h:m:s
#SBATCH -o /rcfs/projects/mantis_shrimp/mantis_shrimp/slurm_scripts/train/o.txt
#SBATCH -e /rcfs/projects/mantis_shrimp/mantis_shrimp/slurm_scripts/train/e.txt



##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

export NCCL_P2P_DISABLE=1

echo
echo "loaded modules"
echo
module list >& _modules.lis_
cat _modules.lis_
/bin/rm -f _modules.lis_
echo
echo limits
echo
limit
echo
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

#clear rdzv directory
rm -r -f /rcfs/projects/mantis_shrimp/mantis_shrimp/torch_rdzv/*

### NOW RUN

cd /rcfs/projects/mantis_shrimp/mantis_shrimp/NOTEBOOKS/

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export LOGLEVEL=INFO #will enable some debug stuff

free_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')

echo Node IP: $head_node_ip
echo free port: $free_port

srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=8 --nproc_per_node=2 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p2_earlyfusion.py --fusion_type=late

echo "All Done!"
