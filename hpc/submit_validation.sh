#!/bin/bash
#
#SBATCH --job-name=geom-econ-val
#SBATCH --output=geom-econ-val-%j.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL

# ── Geometric Economics Out-of-Sample Validation ──
# Runs on SJSU CoE HPC (no GPU needed — pure numpy/scipy)
#
# Usage:
#   1. SSH into coe-hpc1.sjsu.edu
#   2. cd to this directory
#   3. sbatch submit_validation.sh

echo "=========================================="
echo "Geometric Economics Validation"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Load Python
module load python3

# Install dependencies (user-local)
pip install --user numpy scipy matplotlib 2>&1 | tail -3

# Install eris-econ
cd /home/$USER/eris-econ
pip install --user -e . 2>&1 | tail -3

# Run validation
cd /home/$USER/eris-econ/hpc
python run_validation.py \
    --output-dir /home/$USER/eris-econ/hpc/results \
    --n-bootstrap 500

echo ""
echo "=========================================="
echo "Validation complete at $(date)"
echo "Results in: /home/$USER/eris-econ/hpc/results/"
echo "=========================================="
