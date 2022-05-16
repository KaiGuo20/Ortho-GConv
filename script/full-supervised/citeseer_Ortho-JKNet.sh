#!/bin/bash

python full-supervised.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --data citeseer \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 8 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.004 \
    --weight_decay 5e-05 \
    --patience 400 \
    --sampling_percent 0.6 \
    --dropout 0.3 \
    --normalization AugNormAdj \
    --aggrmethod "concat" \
    --T1 4 \
    --T2 4 \
    --group2 1 \
    --weight_beta 0.4 \
    --Ortho True \
    --model JKNet \
    --withloop \

    
