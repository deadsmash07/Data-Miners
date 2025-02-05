#!/usr/bin/env bash
# identify.sh
# Usage:
#   bash identify.sh <path_train_graphs_gspan> <path_train_labels> <path_gspan_binary> <min_support>
#
# Example:
#   bash identify.sh /data/train_graphs.dat /data/train_labels.txt /binaries/gspan 0.05
#
# Explanation of parameters:
#   <path_train_graphs_gspan>  = Path to a gSpan-compatible input file (e.g. "mydata.dat")
#   <path_train_labels>        = Path to the labels of the training graphs (one label per line)
#   <path_gspan_binary>        = Path to the gSpan executable
#   <min_support>              = e.g., 0.05 => 5% minimum support

if [ "$#" -ne 4 ]; then
  echo "Usage: bash identify.sh <path_train_graphs_gspan> <path_train_labels> <path_gspan_binary> <min_support>"
  exit 1
fi

PATH_TRAIN_GRAPHS=$1
PATH_TRAIN_LABELS=$2
PATH_GSPAN_BINARY=$3
MIN_SUPPORT=$4    # e.g. 0.05 => 5%

# Call subgraph_mining.py in identify mode,
# without any explicit "out_subs" argument since gSpan
# will automatically produce <path_train_graphs_gspan>.fp.
python3 subgraph_mining.py \
  --mode identify \
  --graphs "$PATH_TRAIN_GRAPHS" \
  --labels "$PATH_TRAIN_LABELS" \
  --binary "$PATH_GSPAN_BINARY" \
  --min_support "$MIN_SUPPORT"

echo "Identification of discriminative subgraphs complete."
