#!/bin/bash

# Check if all 4 arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: bash q1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>"
    exit 1
fi

# Assign arguments to variables
PATH_APRIORI_EXEC=$1
PATH_FP_EXEC=$2
PATH_DATASET=$3
PATH_OUT=$4

# Check if run.py exists in the current directory
if [ ! -f "run.py" ]; then
    echo "Error: run.py not found in the current directory."
    exit 1
fi

# Run the Python script with the arguments
python3 run.py "$PATH_APRIORI_EXEC" "$PATH_FP_EXEC" "$PATH_DATASET" "$PATH_OUT"
