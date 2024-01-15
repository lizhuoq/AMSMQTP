#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

TIBETAN_DATASET_ROOT="../../data/processed/Tibetan/structured_dataset_v5"
CRA_DATASET_ROOT="../../data/processed/CRA/structured_dataset"
LOG_ROOT="../../logs"
CHECKPONTS_ROOT="../../checkpoints"

for k in {1..100}
do
    echo "iter: $k"
    for i in layer1 layer2 layer3 layer4 layer5
    do
        for j in spatial temporal
        do
            python 1.3_tibetan_cra_structured_ml_exp.py \
            --tibetan_dataset_root $TIBETAN_DATASET_ROOT \
            --cra_dataset_root $CRA_DATASET_ROOT \
            --log_root $LOG_ROOT \
            --checkpoints_root $CHECKPONTS_ROOT \
            --layer $i \
            --split_method $j 
        done
    done
done
