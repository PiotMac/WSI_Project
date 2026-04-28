#!/bin/bash
#SBATCH --job-name=qupath_prep
#SBATCH --time=00:15:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --output=log_qupath_%j.txt
#SBATCH --extra=FORCE_RM_TMPDIR

source /home/piomac5688/env_wsi/bin/activate
SLIDE_ID="128249_M_3_S_MGG_14_03"
unzip -q /lustre/pd03/hpc-ljelen-1692966897/WSI/WSI_patched_png/malignant.zip "malignant/$SLIDE_ID/*" -d $TMPDIR/
cd /home/piomac5688/projekt_wsi
python3 -u lookup/create_gallery.py