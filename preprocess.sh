#!/bin/bash
#SBATCH --job-name=preprocess_wsi
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#SBATCH --output=log_preprocess_%j.txt

source /home/piomac5688/env_wsi/bin/activate

unzip -q /lustre/pd03/hpc-ljelen-1692966897/WSI/WSI_patched_png/malignant.zip -d $TMPDIR/
cd /home/piomac5688/projekt_wsi
python3 -u preprocessing.py