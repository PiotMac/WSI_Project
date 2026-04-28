#!/bin/bash
#SBATCH --job-name=feature_extract
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=log_extract_%j.txt
#SBATCH --extra=FORCE_RM_TMPDIR

source /home/piomac5688/env_wsi/bin/activate
cd /home/piomac5688/projekt_wsi
python3 -u extract_features.py