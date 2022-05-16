#!/bin/bash

python -u full-supervised.py --data cora --layer 2 --weight_decay 1e-4 --T1 1 --T2 2 --group2 1 --model GCNII --Ortho True --weight_beta 0.1 --patience 200 --alpha 0.2

    
