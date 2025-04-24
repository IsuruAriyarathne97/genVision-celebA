#!/bin/bash
#SBATCH -J vAE-v1       # Job name
#SBATCH --output=/sciclone/data10/iahewababarand/genVision-VAE/logs/output.log        # Output file
#SBATCH --error=/sciclone/data10/iahewababarand/genVision-VAE/logs/error.log       # Error file
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 1                   # Number of tasks/cores per node
# SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH -t 48:00:00            # Runtime in hh:mm:ss
#SBATCH --mem=80G

# Run your Python script
python run_notebook.py  > logs/output.txt





