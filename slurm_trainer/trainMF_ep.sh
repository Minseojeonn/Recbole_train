#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J MF_ep
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH --chdir=../
for embedding_size in 32 64; do
                python -m train --model_name MF --dataset_name epinions --learning_rate $lr --embedding_size $embedding_size 
done

python -m notify MF ep
exit 0