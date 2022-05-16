#!/bin/bash

python full-supervised.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --data texas \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 8 \
    --hidden 128 \
    --epoch 40 \
    --lr 0.005 \
    --weight_decay 0.0001 \
    --patience 20 \
    --sampling_percent 0.5 \
    --dropout 0.8 \
    --normalization AugNormAdj \
    --withloop \
    --aggrmethod "concat" \
    --T1 2 \
    --T2 2 \
    --group2 1 \
    --weight_beta 0.1 \
    --Ortho True \
    --model JKNet \
    --withbn
