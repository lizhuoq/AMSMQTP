#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16


for l in {1..5}
do
	for s_m in spatial, temporal
	echo "-------------------------layer: layer$l | split_method: $s_m -------------------------"
	do
		python cra_tibetan.py \
			--layer layer1 \
			--split_method spatial \
			--logs_root ../logs \
			--checkpoints_root ../checkpoints \
			--num_layers 2 \
			--hidden_size 64 \
			--split_method s_m \
			--layer layer$l
	done
done
