#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J LGCN_ml-1m
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH --chdir=../
for layer in 1 2 3
do
    for lr in 0.01 0.001
    do
        for emdim in 32 64
        do
            python -m train --model_name LightGCN --dataset_name ml-1m --n_layers $layer --learning_rate $lr --embedding_size $emdim
        done
    done
done

python -m notify LGCN ml-1m
exit 0