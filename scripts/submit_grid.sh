#!/bin/bash

source /home/piomac5688/env_wsi/bin/activate
cd /home/piomac5688/projekt_wsi

STAINS=("mgg" "hes" "both")
EXTRACTORS=("resnet50" "ctranspath" "uni" "phikon")
MODELS=("gated_attention" "transmil" "clam")

echo "Adding jobs to SLURM queue..."
echo "------------------------------------------------------------------"

for STAIN in "${STAINS[@]}"; do
    for EXT in "${EXTRACTORS[@]}"; do
        for MOD in "${MODELS[@]}"; do
            JOB_NAME="${STAIN}_${EXT}_${MOD}"

            sbatch --job-name=$JOB_NAME scripts/mil_job.sh "$STAIN" "$EXT" "$MOD"

            echo "Added job: $JOB_NAME"

            sleep 1
        done
    done
done

echo "------------------------------------------------------------------"
echo "All jobs were successfully added to the queue."