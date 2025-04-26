#!/usr/bin/env bash
rm -rf task1_d2_train.pt task1_d2_test.pt task1_d2_model.pt preds1_d2.csv

python prepare_task1.py \
    --edges datasets/task1/d2/edges.npy \
    --features datasets/task1/d2/node_feat.npy \
    --labels datasets/task1/d2/label.npy \
    --train-output task1_d2_train.pt \
    --test-output task1_d2_test.pt \
    --test-size 0.2 \
    --seed 42

bash train1_d2.sh task1_d2_train.pt task1_d2_model.pt

bash test1_d2.sh task1_d2_test.pt task1_d2_model.pt preds1_d2.csv

python eval_task1_d2.py \
    --true task1_d2_test.pt \
    --pred preds1_d2.csv
