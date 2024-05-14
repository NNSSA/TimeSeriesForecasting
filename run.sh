#!/bin/bash -l
#SBATCH --job-name=XTY
#SBATCH --time=0-4:00:00
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH -A mkamino1_gpu ## tedwar41_gpu

module load anaconda/2020.07
module load cuda/12.1.0
module load cudnn/8.0.4.30-11.1-linux-x64
conda activate XTY
time python -u run.py