#!/usr/bin/env bash

# identify.sh
# Usage: bash identify.sh <path_train_graphs> <path_train_labels> <path_discriminative_subgraphs>

PATH_TRAIN_GRAPHS=$1
PATH_TRAIN_LABELS=$2
PATH_DISCRIM_SUBS=$3

python3 subgraph_mining.py \
  --mode identify \
  --graphs "$PATH_TRAIN_GRAPHS" \
  --labels "$PATH_TRAIN_LABELS" \
  --out_subs "$PATH_DISCRIM_SUBS"

echo "Identification of discriminative subgraphs complete."
