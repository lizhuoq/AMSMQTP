#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64


for l in {1..5}
do
	for s_m in spatial temporal
	do
		for md in lgbm xgboost xgb_limitdepth catboost rf extra_tree auto
		do
			echo layer$l,$s_m,$md
			python discussion_automl.py \
					--time_budget 2 \
					--iid adversial_validation \
					--layer layer$l \
					--split_method $s_m \
					--logs_root ../discussion/logs \
					--test_results_root ../discussion/test_results \
					--checkpoints_root ../discussion/checkpoints \
					--estimator $md \
					--test_time_budget
		done
	done
done
