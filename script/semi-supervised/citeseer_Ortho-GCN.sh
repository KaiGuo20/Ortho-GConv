#!/bin/bash

python -u train.py --data citeseer --layer 2 --T1 4 --T2 4 --group2 1 --model Ortho_GCN --bias True --Ortho True --weight_beta 0.5 --patience 200

    
