#!/bin/bash
#SBATCH --job-name=extractors
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=log_extractors_%j.txt
#SBATCH --extra=FORCE_RM_TMPDIR

source /home/piomac5688/env_wsi_legacy/bin/activate
cd /home/piomac5688/projekt_wsi

MODELS=("ctranspath")
CATEGORIES=("benignHES" "benignMGG" "malignantHES" "malignantMGG")

for CAT in "${CATEGORIES[@]}"; do
    echo "**********************************************************"
    echo "PROCESSING DATASET: $CAT"
    echo "**********************************************************"

    echo "=========================================================="
    echo "EXTRACTION MODEL: ctranspath FOR DATASET: $CAT"
    echo "=========================================================="

    python3 -u extraction_module.py \
            --model ctranspath \
            --zip_path "/lustre/pd03/hpc-ljelen-1692966897/WSI/WSI_patched_png/${CAT}.zip" \
            --csv_dir "/home/piomac5688/projekt_wsi/csv_filtered_patches/${CAT}" \
            --output_dir "/lustre/pd03/hpc-ljelen-1692966897/WSI_features/${CAT}" \
            --weights "/home/piomac5688/projekt_wsi/ctranspath.pth"
done

echo "CTransPath tested successfully!"