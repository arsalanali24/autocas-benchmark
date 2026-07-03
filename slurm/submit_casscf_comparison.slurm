#!/bin/bash
#SBATCH --job-name=casscf_comp
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-qehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/casscf_comparison_%j.log
#SBATCH --error=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/casscf_comparison_%j.err

# CASSCF comparison: autoCAS active spaces with orbital optimization
# Expected runtime: 8-16 hours (MoCl6 CAS(15,14) is the most expensive)
# QCMaquis is NOT touched — pure PySCF CASSCF only

set -uo pipefail
source ~/.autocas_env.sh || true
set -e

MKL22="/opt/software/pc2/EB-SW/software/imkl/2022.2.1/mkl/2022.2.1/lib/intel64"
GCC11="/opt/software/pc2/EB-SW/software/GCCcore/11.3.0/lib64"
export LD_PRELOAD="${GCC11}/libgomp.so.1:${MKL22}/libmkl_gnu_thread.so.2:${MKL22}/libmkl_core.so.2"
export LD_LIBRARY_PATH="${MKL22}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== CASSCF comparison vs autoCAS CASCI ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "QCMaquis NOT involved — pure PySCF CASSCF only"
echo ""

cd "$HOME/activeml/scripts"
python3 -u run_casscf_comparison.py

echo ""
echo "Finished: $(date)"
echo "Report: casscf_comparison_output/casscf_comparison_report.md"
