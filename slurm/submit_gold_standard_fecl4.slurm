#!/bin/bash
#SBATCH --job-name=gold_fecl4
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-qehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/gold_fecl4_%j.log
#SBATCH --error=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/gold_fecl4_%j.err

# FeCl4^1- gold-standard autoCAS run.
# Key differences vs CrCl6 job (33399406):
#   - Symmetry ENABLED (Td -> D2 Abelian subgroup expected)
#   - Smaller system (4 Cl ligands, tetrahedral vs 6 Cl octahedral)
# Purpose: direct timing comparison to quantify the symmetry benefit.

set -uo pipefail
source ~/.autocas_env.sh || true
set -e

MKL22="/opt/software/pc2/EB-SW/software/imkl/2022.2.1/mkl/2022.2.1/lib/intel64"
GCC11="/opt/software/pc2/EB-SW/software/GCCcore/11.3.0/lib64"

export LD_PRELOAD="${GCC11}/libgomp.so.1:${MKL22}/libmkl_gnu_thread.so.2:${MKL22}/libmkl_core.so.2"
export LD_LIBRARY_PATH="${MKL22}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Gold-standard autoCAS: FeCl4^1- (symmetry enabled) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

cd "$HOME/activeml/scripts"
python3 -u run_gold_standard_fecl4.py

echo ""
echo "Finished: $(date)"
