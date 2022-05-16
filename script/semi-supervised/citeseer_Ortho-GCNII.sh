#!/bin/bash

python -u train.py --data citeseer --layer 32 --hidden 256 --lamda 0.6 --dropout 0.7 --T1 2 --T2 0 --group2 1 --model GCNII  --weight_beta 1 --patience 200 --wd1 0.01 --get acc
    
