#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

python 5_discussion_ensemble_monthly_mean.py
python 5_discussion_ensemble_yearly_mean.py
