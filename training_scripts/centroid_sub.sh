#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=centroid_training
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --output=logs/centroid_out.%j.out
#SBATCH --error=logs/centroid_err.%j.err

module load anaconda3/2021.05

source /home/laird.l/.bashrc
source activate HON_env

python centroid_training.py  --dataset_name even10k_simplex_data --ne 100 --x_agg mean --pooling mean --residual True