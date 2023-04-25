#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=eHON_training
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --output=logs/eHON_out.%j.out
#SBATCH --error=logs/eHON_err.%j.err

module load anaconda3/2021.05
module load cuda/11.7

source /home/laird.l/.bashrc
source activate HON_env2

python eHON_training.py --dataset_name even10k_simplex_data --ne 100 --pooling max --x_agg mean --residual True --use_additional False