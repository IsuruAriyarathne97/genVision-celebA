#!/bin/bash
#SBATCH -J WGAN       # Job name
#SBATCH --output=/sciclone/data10/jrhee01/genVision-celebA/logs/wgan-output.log        # Output file
#SBATCH --error=/sciclone/data10/jrhee01/genVision-celebA/logs/wgan-error.log       # Error file
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 1                   # Number of tasks/cores per node
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH -t 48:00:00            # Runtime in hh:mm:ss
#SBATCH --mem=80G

# Run your Python script
python run_notebook.py  > logs/wgan-output.txt





