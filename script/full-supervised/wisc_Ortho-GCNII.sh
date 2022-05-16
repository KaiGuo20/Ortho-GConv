#!/bin/bash

python -u full-supervised.py --data wisconsin --layer 4 --weight_decay 5e-4 --T1 1 --T2 4 --group2 1 --model GCNII --Ortho True --weight_beta 0.2 --patience 200 --lamda 1

    
