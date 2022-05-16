#!/bin/bash

python train.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --data cora \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 8 \
    --hidden 128 \
    --epochs 400 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --patience 400 \
    --sampling_percent 1 \
    --dropout 0.8 \
    --aggrmethod "concat" \
    --T1 2 \
    --T2 4 \
    --group2 1 \
    --weight_beta 0.2 \
    --Ortho True \
    --model JKNet \
    --normalization AugRWalk --task_type semi \
     \

    
