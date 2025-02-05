#!/bin/bash

# Check if all 4 arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>"
    exit 1
fi

# Assign arguments to variables
PATH_GSPAN_EXEC=$1
PATH_FSG_EXEC=$2
PATH_GASTON_EXEC=$3
PATH_DATASET=$4
PATH_OUT=$5

# Check if run.py exists in the current directory
if [ ! -f "run.py" ]; then
    echo "Error: run.py not found in the current directory."
    exit 1
fi

# Run the Python script with the arguments
python3 run.py "$PATH_GSPAN_EXEC" "$PATH_FSG_EXEC" "$PATH_GASTON_EXEC" "$PATH_DATASET" "$PATH_OUT"
