#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

for l in {1..5}
do
    python history_ml.py \
            --layer layer$l \
            --history_root ../history
done
