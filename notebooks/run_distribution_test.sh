#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16


for i in {1..1000}
do
	for l in {1..5}
	do
		python distribution_test.py \
			--layer layer$l \
			--split_method spatial \
			--seed $i \
			--distribution_root ../distribution_logs
	done
done
