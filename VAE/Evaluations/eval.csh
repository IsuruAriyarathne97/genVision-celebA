#!/bin/csh


#SBATCH -J eval_vqvae
#SBATCH --output=errors/eval_vqvae_%j.out
#SBATCH --error=errors/eval_vqvae_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: `date`"
echo "Number of GPUs: $SLURM_JOB_GPUS"

mkdir -p errors
mkdir -p samples

source ../venv/bin/activate.csh

python eval.py \
    --model_path ../chkpts/vqvae_step40000.pt \
    --data_root ../data \
    --batch 64 \
    --calculate_fid 
