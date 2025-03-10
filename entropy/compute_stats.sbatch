#!/usr/bin/env bash
#
#SBATCH --job-name=train
#SBATCH --time=0-00:05:00
#SBATCH --partition=a100
#SBATCH --qos=pzal_a100
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output="logs/slurm-%j.txt"
#SBATCH --error="logs/slurm-%j.txt"

srun uv run scripts/compute_norm_stats.py --config-name pi0_nomagic_urx_joint_space
