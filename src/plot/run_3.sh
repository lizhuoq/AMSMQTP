#!/bin/bash

for l in {1..5}
do
	for s_m in spatial temporal
	do
        echo layer$l,$s_m
		python 3_train_test.py \
            --iid adversial_validation \
            --layer layer$l \
            --split_method $s_m
	done
done
