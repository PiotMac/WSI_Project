#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/tuning_clam/log_%x_%j.txt

source /home/piomac5688/env_wsi/bin/activate
cd /home/piomac5688/projekt_wsi

STAIN=$1
EXTRACTOR=$2
B=$3
C1=$4

echo "=========================================================="
echo "CLAM TUNING: Stain=[$STAIN] | Ext=[$EXTRACTOR] | B=[$B] | C1=[$C1]"
echo "=========================================================="

python3 -u train_mil.py \
    --stain "$STAIN" \
    --extractor "$EXTRACTOR" \
    --mil_model "clam" \
    --clam_b "$B" \
    --clam_c1 "$C1" \
    --epochs 200 \
    --patience 10
