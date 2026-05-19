#!/bin/bash
#SBATCH --job-name=mil_tune
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/training/%x_%j.out

source /home/piomac5688/env_wsi/bin/activate
cd /home/piomac5688/projekt_wsi

STAIN=$1
EXTRACTOR=$2
MODEL=$3

python3 -u train_mil.py \
    --stain $STAIN \
    --extractor $EXTRACTOR \
    --mil_model $MODEL \
    --epochs 200 \
    --patience 10