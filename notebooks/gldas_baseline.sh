#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16


for l in {1..5}
do
	for s_m in spatial temporal
	do
		python gldas_baseline.py \
			--iid adversial_validation \
			--layer layer$l \
			--split_method $s_m \
			--logs_root ../baseline/logs \
			--test_results_root ../baseline/test_results
	done
done
