#!/bin/bash

python -u full-supervised.py --data pubmed --layer 2 --weight_decay 5e-6 --T1 2 --T2 2 --group2 1 --model Ortho_GCN --bias True --Ortho True --weight_beta 0.5 --patience 200
