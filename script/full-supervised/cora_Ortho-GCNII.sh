#!/bin/bash

python -u full-supervised.py --data cora --layer 4 --weight_decay 1e-4 --T1 1 --T2 4 --group2 1 --model GCNII --Ortho True --weight_beta 0.9 --patience 200 --alpha 0.2 --variant

    
