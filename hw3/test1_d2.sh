#!/usr/bin/env bash

rm -f "$TEMP_TEST_PT" 
# Convert input path to use the right files
DATASET_DIR="$1"
MODEL_PATH="$2"
OUTPUT_PATH="$3"
TEMP_TEST_PT="temp_test_d1.pt"

# Create test .pt file from the dataset
python prepare_task1.py \
    --edges "${DATASET_DIR}/edges.npy" \
    --features "${DATASET_DIR}/node_feat.npy" \
    --labels "${DATASET_DIR}/label.npy" \
    --output "$TEMP_TEST_PT" \

# Run test using the prepared .pt file
python src/test1_d2.py --input "$TEMP_TEST_PT" --model "$MODEL_PATH" --output "$OUTPUT_PATH"

# Clean up temporary files
