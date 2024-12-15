#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J ENMF_ml-1m
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH --chdir=../
for negative_socre in 0.01 0.02 
do
    for dropout_rate in 0.1 0.3 0.5 0.7
    do
        for mlp_embedding_size in 32 64 
        do
            for lr in 0.01 0.001
            do
                python -m train --model_name ENMF --dataset_name ml-1m --learning_rate $lr --embedding_size $mlp_embedding_size --dropout_prob $dropout_rate --negative_score $negative_socre 
            done
        done
    done
done

python -m notify ENMF ml-1m
exit 0