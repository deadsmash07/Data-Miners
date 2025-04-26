#!/usr/bin/env bash
rm -rf task2_train.pt task2_test.pt task2_model.pt preds2.csv

python prepare_task2.py \
    --data_dir datasets/task2 \
    --train-output task2_train.pt \
    --test-output task2_test.pt

bash train2.sh task2_train.pt task2_model.pt
bash test2.sh task2_test.pt task2_model.pt preds2.csv
python eval_task2.py \
    --true task2_test.pt \
    --pred preds2.csv