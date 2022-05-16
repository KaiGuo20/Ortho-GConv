#!/bin/bash

python train.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --data pubmed \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 8 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.001 \
    --weight_decay 0.005 \
    --patience 400 \
    --sampling_percent 0.7 \
    --dropout 0.8 \
    --T1 2 \
    --T2 4 \
    --group2 2 \
    --weight_beta 0.4 \
    --Ortho True \
    --model JKNet \
    --normalization AugNormAdj --task_type semi \
     \
    
