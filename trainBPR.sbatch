#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J BPR

python -m train --model_name LightGCN --dataset_name ml-100k

exit 0