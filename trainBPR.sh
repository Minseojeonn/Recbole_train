#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J BPR_yelp2018

for embed_size in 32, 64, 128
do
    for lr in 0.001, 0.0001, 0.00001
    do
        python -m train --model_name BPR --dataset_name yelp2018 --learning_rate $lr --embedding_size $embed_size
    done
done

python -m notify BPR yelp2018
exit 0