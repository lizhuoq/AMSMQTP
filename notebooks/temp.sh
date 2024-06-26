#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

for l in {1..5}
do
	for s_m in spatial temporal
	do
		python cra_tibetan_ml_temp.py \
			--time_budget 600 \
			--iid correct_adversarial_validation \
			--layer layer$l \
			--split_method $s_m \
			--logs_root ../logs \
			--test_results_root ../test_results \
			--checkpoints_root ../checkpoints
	done
done
