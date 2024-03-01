#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

python gssm_baseline.py \
	--iid adversial_validation \
	--layer layer1 \
	--split_method spatial \
	--logs_root ../baseline/logs \
	--test_results_root ../baseline/test_results
        
