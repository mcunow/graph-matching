#!/bin/bash


python save_system_info.py

sudo apt-get update && apt-get install -y ffmpeg

source ~/.bashrc

# Function to install Miniconda
install_miniconda () {
    # Create directory for Miniconda installation
    mkdir -p ~/miniconda3
    # Download the latest Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    # Run the installer silently
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    # Remove the installer script
    rm -rf ~/miniconda3/miniconda.sh
    # Initialize conda for bash and zsh
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh
}

# Check if Miniconda is already installed, if not, install it
if [[ ! -d ~/miniconda3 ]]; then
    install_miniconda > /dev/null 2>&1
fi

# Source the conda.sh script to enable conda in the current shell session
source ~/miniconda3/etc/profile.d/conda.sh

# Check if the Conda environment "new_env" already exists
if ! conda env list | grep -q "^new_env\s"; then
    # Check if environment.yml exists
    if [[ ! -f environment.yml ]]; then
        echo "environment.yml file not found!"
        exit 1
    fi

    # Create the Conda environment with the name "new_env"
    conda env create -n new_env -f environment.yml
fi

# Install ipykernel in the "new_env" environment and register it with Jupyter
conda run -n new_env conda install -y ipykernel
conda run -n new_env python -m ipykernel install --user --name=new_env --display-name "Python (new_env)"

# Save environment
conda env export -n new_env -f new_env_export.yml

conda run -n new_env pip install ffmpeg-python
