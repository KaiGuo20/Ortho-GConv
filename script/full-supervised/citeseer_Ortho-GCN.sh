#!/bin/bash

python -u full-supervised.py --data citeseer --layer 2 --weight_decay 1e-4 --T1 1 --T2 1 --group2 1 --model Ortho_GCN --bias True --Ortho True --weight_beta 0.1 --patience 200
