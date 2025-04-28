#!/usr/bin/env bash

TEMP_TEST_PT="temp_train_d1.pt"
rm -f "$TEMP_TEST_PT" 
# Convert input path to use the right files
DATASET_DIR="$1"
MODEL_PATH="$2"
OUTPUT_PATH="$3"

python prepare_task2.py \
    --data_dir "${DATASET_DIR}" \
    --output "$TEMP_TEST_PT" \

# Run test using the prepared .pt file
python src/test2.py --input "$TEMP_TEST_PT" --model "$MODEL_PATH" --output "$OUTPUT_PATH"

# Clean up temporary files
