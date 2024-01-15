#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64


for l in {1..5}
do
	for s_m in spatial temporal
	do
		python discussion_rf.py \
				--iid adversial_validation \
				--time_budget 2400 \
				--layer layer$l \
				--split_method $s_m \
				--logs_root ../discussion/logs \
				--test_results_root ../discussion/test_results
	done
done
