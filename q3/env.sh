#!/usr/bin/env bash
# env.sh : Minimal environment setup for Q3

# (Adapt for your HPC if needed, e.g., module load python/3.10)
# or create and activate a virtual environment:
# python3 -m venv dm
# source dm/bin/activate

# Install required Python packages
pip install --user numpy networkx scikit-learn

echo "Environment set up. Python 3.10 and necessary libraries installed (numpy, networkx, scikit-learn)."
