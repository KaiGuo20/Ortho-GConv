#!/bin/bash

python -u train.py --data cora --layer 2 --T1 4 --T2 4 --group2 1 --model Ortho_GCN --bias Ture --Ortho True --weight_beta 0.5 --patience 200 --wd1 1e-4 --gama 0.0005
