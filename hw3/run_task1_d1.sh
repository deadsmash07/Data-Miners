#!/usr/bin/env bash
rm -rf task1_d1_train.pt task1_d1_test.pt task1_d1_model.pt preds1_d1.csv

python prepare_task1.py \
    --edges datasets/task1/d1/edges.npy \
    --features datasets/task1/d1/node_feat.npy \
    --labels datasets/task1/d1/label.npy \
    --train-output task1_d1_train.pt \
    --test-output task1_d1_test.pt \
    --test-size 0.2 \
    --seed 42

bash train1_d1.sh task1_d1_train.pt task1_d1_model.pt

bash test1_d1.sh task1_d1_test.pt task1_d1_model.pt preds1_d1.csv

python eval_task1_d1.py \
    --true task1_d1_test.pt \
    --pred preds1_d1.csv

