#!/bin/bash

python -u full-supervised.py --data cornell --layer 4 --weight_decay 1e-3 --T1 2 --T2 4 --group2 1 --model Ortho_GCN --bias True --Ortho True --weight_beta 0.2 --patience 20

