#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J LightGCN

python -m train --model_name BPR --dataset_name ml-100k

exit 0