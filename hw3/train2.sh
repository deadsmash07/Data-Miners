#!/usr/bin/env bash

TEMP_TRAIN_PT="temp_train_d1.pt"
rm -f "$TEMP_TRAIN_PT"
# Convert input path to use the right files
DATASET_DIR="$1"
OUTPUT_PATH="$2"

# Create .pt files from the dataset
python prepare_task2.py \
    --data_dir "${DATASET_DIR}" \
    --output "$TEMP_TRAIN_PT" \

# Train model using the prepared .pt file
python src/train2.py --input "$TEMP_TRAIN_PT" --output "$OUTPUT_PATH"
