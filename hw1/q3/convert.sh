
if [ "$#" -ne 3 ]; then
  echo "Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>"
  exit 1
fi

PATH_GRAPHS=$1
PATH_DISCRIMINATIVE_SUBGRAPHS=$2
PATH_FEATURES=$3

python3 convert.py \
  --graphs "$PATH_GRAPHS" \
  --path_discriminative_subgraphs "$PATH_DISCRIMINATIVE_SUBGRAPHS" \
  --out_features "$PATH_FEATURES"

echo "Conversion complete. Feature matrix saved to $PATH_FEATURES."
