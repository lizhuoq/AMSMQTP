#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64


for l in {1..5}
do
	python inference_discussion_ensemble.py \
		--layer layer$l \
		--output_root ../data/discussion/output_ensemble
done
