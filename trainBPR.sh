#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J BPR_ml-100k

for embed_size in 32, 64, 128
do
    for lr in 0.001, 0.0001, 0.00001
    do
        python -m train --model_name BPR --dataset_name ml-100k --learning_rate $lr --embedding_size $embed_size
    done
done

python -m notify BPR ml-100k
exit 0