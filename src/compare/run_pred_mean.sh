#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

python 0_pred_monthly_mean.py
python 0_pred_yearly_mean.py
