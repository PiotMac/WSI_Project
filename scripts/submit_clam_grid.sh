#!/bin/bash

source /home/piomac5688/env_wsi/bin/activate
cd /home/piomac5688/projekt_wsi

STAINS=("mgg")
EXTRACTORS=("uni")

B_VALUES=(8 16 32 64 128)
C1_VALUES=(0.3 0.5 0.7 0.9)

echo "=========================================================="
echo "STARTING CLAM HYPERPARAMETER GRID SEARCH"
echo "=========================================================="

for STAIN in "${STAINS[@]}"; do
    for EXT in "${EXTRACTORS[@]}"; do
        for B in "${B_VALUES[@]}"; do
            for C1 in "${C1_VALUES[@]}"; do

                JOB_NAME="CLAM_${STAIN}_${EXT}_B${B}_C${C1}"

                sbatch --job-name="$JOB_NAME" scripts/run_clam.sh "$STAIN" "$EXT" "$B" "$C1"

                sleep 1

            done
        done
    done
done
