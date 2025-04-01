#!/bin/bash
# Usage: ./solution.sh <absolute_path_to_graph> <absolute_output_file_path> <k> <#_random_instances>
GRAPH_FILE="$1"
OUTPUT_FILE="$2"
K="$3"
NUM_SIMULATIONS="$4"

module purge
module load compiler/gcc/11.2.0
g++ src/solution.cpp -o solution -O2 -fopenmp
time OMP_NUM_THREADS=12 ./solution "$GRAPH_FILE" "$OUTPUT_FILE" "$K" "$NUM_SIMULATIONS"
