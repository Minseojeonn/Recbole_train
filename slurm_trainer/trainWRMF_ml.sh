#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J WRMF_ml
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH --chdir=../
for embedding_size in 32 64; do
    for aplha in 20; do
        for lambda in 100 200 400; do
                python -m train --model_name WRMF --dataset_name ml-1m --learning_rate $lr --embedding_size $embedding_size 
        done
    done
done

python -m notify WRMF ml
exit 0