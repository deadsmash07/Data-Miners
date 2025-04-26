#!/bin/bash

# Install micromamba if not already installed
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Create the environment
micromamba create -f /kaggle/working/environment.yml -y -n myenv

# Activate environment (setup path)
prefix=/usr/local/envs/myenv
export CONDA_PREFIX=$prefix
export PATH=$prefix/bin:$PATH
export PYTHONNOUSERSITE=1

# Install IPython kernel so the notebook can use the environment
micromamba run -n myenv python -m pip install ipykernel
micromamba run -n myenv python -m ipykernel install --user --name myenv --display-name "Python (myenv)"

echo "Environment setup complete!"
