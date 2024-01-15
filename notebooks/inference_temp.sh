#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16


for l in {1..5}
do
	python inference_ml_temp.py \
		--layer layer$l \
		--output_root ../data/output
done
