#!/bin/bash

#SBATCH --time=240:00:00
#SBATCH --job-name=Solar_GPU2
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/Solar_output2.log
#SBATCH --error=slurm/Solar_error2.log

python train_MLMT_1.0.py