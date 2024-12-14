#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J BPR_ml-1m
#SBATCH -o slurm_logs/%x-%j.out

for embed_size in 32, 64
do
    for lr in 0.01, 0.001
    do
        python -m train --model_name BPR --dataset_name ml-1m --learning_rate $lr --embedding_size $embed_size
    done
done

python -m notify BPR ml-1m
exit 0