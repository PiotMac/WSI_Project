#!/bin/bash
#SBATCH --job-name=qupath_prep
#SBATCH --time=00:15:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null

source /home/piomac5688/env_wsi/bin/activate
SLIDE_ID="507754_M_3_S_MGG_22_03"
unzip -q /lustre/pd03/hpc-ljelen-1692966897/WSI/WSI_patched_png/malignant.zip "malignant/$SLIDE_ID/*" -d $TMPDIR/
cd /home/piomac5688/projekt_wsi
python3 -u lookup/create_gallery.py