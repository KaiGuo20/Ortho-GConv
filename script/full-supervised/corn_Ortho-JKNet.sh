#!/bin/bash

python full-supervised.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --data cornell \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 8 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.005 \
    --weight_decay 0.0001 \
    --patience 400 \
    --sampling_percent 0.5 \
    --dropout 0.8 \
    --normalization AugNormAdj \
    --withloop \
    --T1 2 \
    --T2 4 \
    --group2 1 \
    --weight_beta 1 \
    --Ortho True \
    --model JKNet \
    --aggrmethod "concat" \
