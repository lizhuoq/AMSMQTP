#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

python 4_discussion_best_monthly_mean.py
python 4_discussion_best_yearly_mean.py
