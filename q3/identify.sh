#!/usr/bin/env bash
# identify.sh
# Usage:
#   bash identify.sh <path_train_graphs> <path_train_labels> <path_discriminative_subgraphs> <path_miner_binary> <miner_choice> <min_support>
#
# e.g.:
#   bash identify.sh /data/train_graphs.txt /data/train_labels.txt /output/subs.txt /binaries/gspan gspan 0.05

if [ "$#" -ne 6 ]; then
  echo "Usage: bash identify.sh <path_train_graphs> <path_train_labels> <path_discriminative_subgraphs> <path_miner_binary> <miner_choice> <min_support>"
  exit 1
fi

PATH_TRAIN_GRAPHS=$1
PATH_TRAIN_LABELS=$2
PATH_DISCRIM_SUBS=$3
PATH_MINER_BINARY=$4
MINER_CHOICE=$5    # "gspan" or "gaston"
MIN_SUPPORT=$6     # e.g., 0.05 for 5% support

python3 subgraph_mining.py \
  --mode identify \
  --graphs "$PATH_TRAIN_GRAPHS" \
  --labels "$PATH_TRAIN_LABELS" \
  --out_subs "$PATH_DISCRIM_SUBS" \
  --binary "$PATH_MINER_BINARY" \
  --miner_choice "$MINER_CHOICE" \
  --min_support "$MIN_SUPPORT"

echo "Identification of discriminative subgraphs complete. Saved to $PATH_DISCRIM_SUBS."
