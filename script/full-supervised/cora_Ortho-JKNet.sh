#!/bin/bash

python full-supervised.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --data cora \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 8 \
    --hidden 128 \
    --epoch 100 \
    --lr 0.008 \
    --weight_decay 0.0005 \
    --patience 400 \
    --sampling_percent 0.2 \
    --dropout 0.8 \
    --aggrmethod "concat" \
    --T1 2 \
    --T2 4 \
    --group2 2 \
    --weight_beta 0.2 \
    --Ortho True \
    --model JKNet \
    --normalization AugNormAdj \
    \

