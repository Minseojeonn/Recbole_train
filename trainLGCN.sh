#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J LGCN_yelp2018

for layer in 1, 2, 3
do
    for lr in 0.001, 0.0001, 0.00001 
    do
        for emdim in 32, 64, 128
        do
            python -m train --model_name LightGCN --dataset_name yelp2018 --n_layers $layer --learning_rate $lr --embedding_size $emdim
        done
    done
done

python -m notify LGCN yelp2018
exit 0