#!/usr/bin/env bash
#
#SBATCH --job-name=train
#SBATCH --time=0-20:00:00
#SBATCH --partition=a100
#SBATCH --qos=pzal_a100
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output="logs/slurm-%j.txt"
#SBATCH --error="logs/slurm-%j.txt"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 srun uv run scripts/train.py pi0_nomagic_urx_joint_space --exp-name=finetune__pi0_nomagic_urx_joint_space --overwrite
