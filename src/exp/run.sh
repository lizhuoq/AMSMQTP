#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

DATASET_ROOT="../../data/processed/Tibetan/structured_dataset_v5"
LOG_ROOT="../../logs"
CHECKPONTS_ROOT="../../checkpoints"

for k in {1..100}
do
    echo "iter: $k"
    for i in layer1 layer2 layer3 layer5
    do
        for j in spatial temporal
        do
            python 1.1_tibetan_structured_ml_exp.py \
            --dataset_root $DATASET_ROOT \
            --log_root $LOG_ROOT \
            --checkpoints_root $CHECKPONTS_ROOT \
            --layer $i \
            --split_method $j 

            python 1.1_tibetan_structured_ml_exp.py \
            --dataset_root $DATASET_ROOT \
            --log_root $LOG_ROOT \
            --checkpoints_root $CHECKPONTS_ROOT \
            --layer $i \
            --split_method $j \
            --use_era5_mean_std
        done
    done
done
