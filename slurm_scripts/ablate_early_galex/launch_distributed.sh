#!/bin/bash

#SBATCH -A mantis_shrimp
#SBATCH -J ABLATE_GALEX    # create a short name for your job
#SBATCH -p a100,a100_80
#SBATCH --nodes=2 #number of nodes (of machines)
###SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8            # number of gpus per task/process 
#SBATCH --exclusive
#SBATCH -t 96:0:0 #h:m:s
#SBATCH -o ./o.txt
#SBATCH -e ./e.txt



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
module load cuda/12.1
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate torch2.3_new

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

srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4_ablate.py --fusion_type=early --initial_lr=0.0003 --weight_decay=0.0000001 --hidden_width=2048 --n_classes=400 --random_seed=1 --n_epochs=150 --unwise=True

echo "All Done!"
