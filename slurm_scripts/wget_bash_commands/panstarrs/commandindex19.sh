#!/bin/sh
#SBATCH -A mantis_shrimp    #account name-- this might be a WP
#SBATCH -t 96:0:0    #time h:m:s
#SBATCH -N 1    #Number of nodes
#SBATCH -n 10    #Number of cores
#SBATCH -J panstarrs19    #name of the job
#SBATCH -o "./out/%j.o"    #name of the output file
#SBATCH -e "./out/%j.e"    #name of the error file
#SBATCH -p shared

cd ~

cd /rcfs/projects/mantis_shrimp/mantis_shrimp/data/wget_bash_commands/panstarrs/

bash command_index19.sh
