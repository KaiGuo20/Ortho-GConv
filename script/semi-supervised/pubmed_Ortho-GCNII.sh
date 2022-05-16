#!/bin/bash

python -u train.py --data pubmed --layer 16 --T1 2 --T2 4 --group2 1 --model GCNII --Ortho True --weight_beta 0.4 --patience 200 --lamda 0.4 --dropout 0.5 --hidden 256 --get acc
