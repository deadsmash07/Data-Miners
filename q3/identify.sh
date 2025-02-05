#!/usr/bin/env bash
# identify.sh
# Usage:
#   bash identify.sh <path_train_graphs> <path_train_labels> <path_discriminative_subgraphs> <path_miner_binary> <min_support>
#
# e.g.:
#   bash identify.sh /data/train_graphs_gspan.dat /data/train_labels.txt /output/subs.txt /binaries/gspan 0.05
#
# <path_train_graphs>          = absolute filepath to the gSpan-compatible input file
# <path_train_labels>          = absolute filepath to the labels of training graphs
# <path_discriminative_subs>   = absolute filepath where the final subgraphs will be stored
# <path_miner_binary>          = absolute filepath to your gSpan executable
# <min_support>                = e.g. 0.05 for 5% minimum support

if [ "$#" -ne 5 ]; then
  echo "Usage: bash identify.sh <path_train_graphs> <path_train_labels> <path_discriminative_subgraphs> <path_miner_binary> <min_support>"
  exit 1
fi

PATH_TRAIN_GRAPHS=$1
PATH_TRAIN_LABELS=$2
PATH_DISCRIM_SUBS=$3
PATH_MINER_BINARY=$4
MIN_SUPPORT=$5  # e.g., 0.05 => 5%

python3 subgraph_mining.py \
  --mode identify \
  --graphs "$PATH_TRAIN_GRAPHS" \
  --labels "$PATH_TRAIN_LABELS" \
  --out_subs "$PATH_DISCRIM_SUBS" \
  --binary "$PATH_MINER_BINARY" \
  --min_support "$MIN_SUPPORT"

echo "Identification of discriminative subgraphs complete. Saved to $PATH_DISCRIM_SUBS."
