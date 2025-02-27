#!/bin/bash

python train_soup3.py \
    --feats_size ? \
    --num_classes ? \
    --lr ? \
    --weight_decay ? \
    --dropout_node ? \
    --dropout_patch ? \
    --non_linearity ? \
    --q_n ? \
    --average ? \
    --dataset ? \
    --model maxmil \
    --init_seed ? \
    --nb_models ? \
    --init_type ? \
    --num_epochs 40 1>~/out?.txt 2>~/error?.txt 

echo "done"
