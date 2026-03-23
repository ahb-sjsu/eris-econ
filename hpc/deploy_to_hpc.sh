#!/bin/bash
# Deploy eris-econ to SJSU CoE HPC
#
# Usage: bash deploy_to_hpc.sh <SJSU_ID>
# Example: bash deploy_to_hpc.sh abond

SJSU_ID=${1:?"Usage: bash deploy_to_hpc.sh <SJSU_ID>"}
HPC_HOST="coe-hpc1.sjsu.edu"

echo "Deploying eris-econ to ${SJSU_ID}@${HPC_HOST}..."

# Sync the repo (excluding large/unnecessary files)
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.eggs' \
    --exclude '*.egg-info' \
    --exclude 'htmlcov' \
    --exclude '.mypy_cache' \
    --exclude '.ruff_cache' \
    --exclude 'node_modules' \
    /c/source/eris-econ/ \
    ${SJSU_ID}@${HPC_HOST}:/home/${SJSU_ID}/eris-econ/

echo ""
echo "Deployed. Now SSH in and submit:"
echo "  ssh ${SJSU_ID}@${HPC_HOST}"
echo "  cd eris-econ/hpc"
echo "  sbatch submit_validation.sh"
