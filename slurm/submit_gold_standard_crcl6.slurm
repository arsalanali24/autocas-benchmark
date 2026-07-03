#!/bin/bash
#SBATCH --job-name=gold_crcl6
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-qehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/gold_crcl6_%j.log
#SBATCH --error=/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/gold_crcl6_%j.err

# ----------------------------------------------------------------------
# submit_gold_standard_crcl6.slurm
#
# Runs run_gold_standard_crcl6.py (the REAL scine_autocas LargeCasWorkflow
# via the patched PySCF interface) as a proper SLURM batch job, not
# interactively on a login node -- the interactive attempt was killed
# silently, almost certainly by login-node resource enforcement, right
# at the UHF step.
#
# WHAT IS REUSED FROM EARLIER SESSIONS, AND WHY IT'S STILL TRUSTWORTHY:
#   - The LD_PRELOAD of imkl/2022.2.1 before Python starts. This fixes a
#     missing `mkl_sparse_optimize_bsr_trsm_i8` symbol in MKL 2021.4 on
#     production DMRG. This was specifically confirmed via the H2 DMRG
#     smoke test completing cleanly with physically correct entropies
#     (s1 = [0.323, 0.323]) -- an infrastructure-level result, independent
#     of which active-space algorithm runs on top of it.
#   - The SLURM account/partition (hpc-prf-qehpc / normal) -- these are
#     just cluster bookkeeping, not a methodology claim.
#
# WHAT IS NOT REUSED, AND WHY:
#   - Any runtime/walltime estimate from earlier sessions ("8 hours is
#     plenty", "~30 min for octahedral halides"). Those numbers came
#     from a different, ad hoc, hand-rolled script (autocas_mp2no_
#     benchmark.py / autocas_original.py) that has since been abandoned
#     in favour of the real scine_autocas package. This is the first
#     batch run of the actual validated algorithm -- 24h is a generous
#     ceiling, not a confident estimate. If it finishes much faster,
#     that's good news, not evidence the estimate was right.
#
# This script does NOT run the abandoned ad hoc scripts. It runs the
# real package, end to end, exactly as already validated interactively
# (UHF correct, ECP correct, auto-selected 38-orbital valence space)
# up to the point the login node killed it.
# ----------------------------------------------------------------------

set -uo pipefail
source ~/.autocas_env.sh || true
set -e

MKL22="/opt/software/pc2/EB-SW/software/imkl/2022.2.1/mkl/2022.2.1/lib/intel64"
GCC11="/opt/software/pc2/EB-SW/software/GCCcore/11.3.0/lib64"

export LD_PRELOAD="${GCC11}/libgomp.so.1:${MKL22}/libmkl_gnu_thread.so.2:${MKL22}/libmkl_core.so.2"
export LD_LIBRARY_PATH="${MKL22}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Gold-standard autoCAS: CrCl6^3- (real scine_autocas package) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "LD_PRELOAD: $LD_PRELOAD"
echo "Start: $(date)"
echo ""

cd "$HOME/activeml/scripts"

# -u: unbuffered stdout/stderr, so SLURM's own .log file updates live
# instead of going silent until the process exits (the exact problem
# that made the interactive run look like a hang with no information).
python3 -u run_gold_standard_crcl6.py

echo ""
echo "Finished: $(date)"
