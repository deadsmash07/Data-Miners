#!/usr/bin/env bash

rm -f "$TEMP_TRAIN_PT"
# Convert input path to use the right files
DATASET_DIR="$1"
OUTPUT_PATH="$2"
TEMP_TRAIN_PT="temp_train_d1.pt"

# Create .pt files from the dataset
python src/prepare_task1.py \
    --edges "${DATASET_DIR}/edges.npy" \
    --features "${DATASET_DIR}/node_feat.npy" \
    --labels "${DATASET_DIR}/label.npy" \
    --output "$TEMP_TRAIN_PT" \

# Train model using the prepared .pt file
python src/train1_d1.py --input "$TEMP_TRAIN_PT" --output "$OUTPUT_PATH"
