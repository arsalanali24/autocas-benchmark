#!/bin/bash
#SBATCH --job-name=rh_nh3_exp
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-qehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/explore_rhnh3_%j.log
#SBATCH --error=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/explore_rhnh3_%j.err

# Exploratory autoCAS: Rh(NH3)6^3+
# M=100 sweeps=3 — reduced quality for faster active space identification
# Expected runtime: 1-3 hours (vs 10+ hours at gold-standard quality)

set -uo pipefail
source ~/.autocas_env.sh || true
set -e

MKL22="/opt/software/pc2/EB-SW/software/imkl/2022.2.1/mkl/2022.2.1/lib/intel64"
GCC11="/opt/software/pc2/EB-SW/software/GCCcore/11.3.0/lib64"
export LD_PRELOAD="${GCC11}/libgomp.so.1:${MKL22}/libmkl_gnu_thread.so.2:${MKL22}/libmkl_core.so.2"
export LD_LIBRARY_PATH="${MKL22}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Exploratory autoCAS: Rh(NH3)6^3+ ==="
echo "DMRG quality: M=100, sweeps=3"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo ""

cd "$HOME/activeml/scripts"
python3 -u run_explore_rh_nh3.py

echo ""
echo "Finished: $(date)"
echo "Results: explore_rh_nh3/autocas_project/"
