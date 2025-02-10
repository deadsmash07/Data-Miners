

if [ "$#" -ne 3 ]; then
  echo "Usage: bash identify.sh <path_train_graphs> <path_train_labels> <path_discriminative_subgraphs>"
  exit 1
fi

PATH_TRAIN_GRAPHS=$1
PATH_TRAIN_LABELS=$2
PATH_DISCRIMINATIVE_SUBGRAPHS=$3

python3 identify.py \
  --graphs "$PATH_TRAIN_GRAPHS" \
  --labels "$PATH_TRAIN_LABELS" \
  --path_discriminative_subgraphs "$PATH_DISCRIMINATIVE_SUBGRAPHS" 
echo "Identification of discriminative subgraphs complete."
