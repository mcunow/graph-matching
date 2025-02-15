#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Missing argument. You must specify the permutation-invariant loss. The options are: "Optimal", "Top10", "Top50", "Top100", "GNN", "Statistics""
    exit 1  # Exit with a non-zero status to indicate an error
fi

USE_WANDB="True"  # Set to False to disable WandB usage

#TODO
source ~/miniconda3/etc/profile.d/conda.sh
CONDA_NAME="Graph-Matching"
conda activate $CONDA_NAME

# Read the API key from the txt file (if WANDB is used)
if [ "$USE_WANDB" = "True" ]; then
    export USE_WANDB
    export WANDB_MODE=online
    export WANDB_DIR=/tmp #This prevents that a folder is created every time the script is run.

    WANDB_KEY=$(cat ../src/wandb_key.txt)
    export WANDB_KEY
fi

cd ../src


python3 main.py --perm_inv_loss "$1" || { echo "Python script execution failed"; exit 1; }

conda deactivate

