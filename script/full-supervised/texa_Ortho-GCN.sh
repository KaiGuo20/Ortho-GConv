#!/bin/bash

python -u full-supervised.py --data texas --layer 2 --weight_decay 1e-3 --T1 1 --T2 1 --group2 1 --model Ortho_GCN --bias False --Ortho True --weight_beta 0.5 --patience 200

