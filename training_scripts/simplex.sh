#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --job-name=simplex
#SBATCH --mem=20GB
#SBATCH --ntasks=1
#SBATCH --output=logs/simplex_out.%j.out
#SBATCH --error=logs/simplex_err.%j.err

module load anaconda3/2021.05

source /home/laird.l/.bashrc
source activate HON_env

python simplex_data.py