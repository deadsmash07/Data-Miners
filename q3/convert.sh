#!/usr/bin/env bash
# convert.sh
# Usage:
#   bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

if [ "$#" -ne 3 ]; then
  echo "Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>"
  exit 1
fi

PATH_GRAPHS=$1
PATH_DISCRIM_SUBS=$2
PATH_FEATURES=$3

python3 subgraph_mining.py \
  --mode convert \
  --graphs "$PATH_GRAPHS" \
  --in_subs "$PATH_DISCRIM_SUBS" \
  --out_features "$PATH_FEATURES"

echo "Conversion complete. Feature matrix saved to $PATH_FEATURES."
