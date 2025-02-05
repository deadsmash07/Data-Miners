#!/usr/bin/env bash

# env.sh
# Example environment setup script for HPC or local machine.
# We assume Python 3.10 is available. Adapt as needed.

# For HPC, you might do:
# module load python/3.10

# Otherwise, for a local conda environment:conda install networkx scikit-learn numpy scipy
pip install gspan_miner

# If conda-forge has gspan_miner, you can do:
# conda install -c conda-forge gspan_miner networkx scikit-learn numpy scipy
# Otherwise, use pip:
pip install --user gspan_miner networkx scikit-learn numpy scipy

echo "Environment setup complete."
