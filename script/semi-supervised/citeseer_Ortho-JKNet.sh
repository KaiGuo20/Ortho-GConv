#!/bin/bash

python train.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --data citeseer \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 8 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.006 \
    --weight_decay 0.001 \
    --patience 400 \
    --sampling_percent 0.5 \
    --dropout 0.8 \
    --normalization AugNormAdj --task_type semi \
    --T1 4 \
    --T2 2 \
    --group2 1 \
    --weight_beta 0.4 \
    --Ortho True \
    --model JKNet \
    --withloop \
    
