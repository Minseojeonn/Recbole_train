#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J LGCN_epinions
#SBATCH -o slurm_logs/%x-%j.out
for layer in 1, 2, 3
do
    for lr in 0.01, 0.001
    do
        for emdim in 32, 64
        do
            python -m train --model_name LightGCN --dataset_name epinions --n_layers $layer --learning_rate $lr --embedding_size $emdim
        done
    done
done

python -m notify LGCN epinions
exit 0