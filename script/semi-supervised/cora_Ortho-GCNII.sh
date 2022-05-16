#!/bin/bash

python -u train.py --data cora --layer 64 --T1 2 --T2 5 --group2 2 --model GCNII --Ortho True --weight_beta 0.4 --patience 200
