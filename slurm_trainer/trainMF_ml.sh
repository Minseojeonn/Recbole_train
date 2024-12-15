#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J MF_ml
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH --chdir=../
for embedding_size in 32 64; do
    python -m train --model_name MF --dataset_name ml-1m --embedding_size $embedding_size --valid_metric GAUC
done 

python -m notify MF ml
exit 0