#!/bin/bash
#SBATCH -A mantis_shrimp
#SBATCH -J MS_2_2   # create a short name for your job
#SBATCH -p a100,a100_80
#SBATCH --nodes=2 #number of nodes (of machines)
#SBATCH --gres=gpu:8            # number of gpus per task/process
#SBATCH --exclusive
#SBATCH -t 48:0:0 #h:m:s
#SBATCH -o ./o.txt
#SBATCH -e ./e.txt

echo "Environment Variables"
echo
printenv

#clear rdzv directory
rm -r -f /rcfs/projects/mantis_shrimp/mantis_shrimp/torch_rdzv/*

# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc 
source ~/.bashrc

module purge

module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate torch2.3_new
module load cuda/12.1

cd /rcfs/projects/mantis_shrimp/mantis_shrimp/NOTEBOOKS/

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export LOGLEVEL=INFO #will enable some debug stuff

free_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')

echo Node IP: $head_node_ip
echo free port: $free_port


#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.00001 --hidden_width=1024 --n_classes=400 --random_seed=1

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.0001 --hidden_width=1024 --n_classes=400 --random_seed=1

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.001 --hidden_width=1024 --n_classes=400 --random_seed=1

srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.00001 --hidden_width=1600 --n_classes=400 --random_seed=1

srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.00001 --hidden_width=2048 --n_classes=400 --random_seed=1

srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.00001 --hidden_width=2048 --n_classes=500 --random_seed=1

#XXX

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0003 --weight_decay=0.00001 --hidden_width=2048 --n_classes=400 --random_seed=1

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0003 --weight_decay=0.000001 --hidden_width=2048 --n_classes=400 --random_seed=1

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0003 --weight_decay=0.0000001 --hidden_width=2048 --n_classes=400 --random_seed=1

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0005 --weight_decay=0.00001 --hidden_width=2048 --n_classes=400 --random_seed=1

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.000001 --hidden_width=2048 --n_classes=400 --random_seed=1

#srun torchrun --rdzv_endpoint=$head_node_ip:$free_port --nnodes=2 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d  --max-restarts=2 ./train_V1p4.py --fusion_type=late --initial_lr=0.0001 --weight_decay=0.0000001 --hidden_width=2048 --n_classes=400 --random_seed=1


