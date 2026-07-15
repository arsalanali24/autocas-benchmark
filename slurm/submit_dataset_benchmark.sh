#!/bin/bash
# submit_dataset_benchmark.sh
#
# Submits one SLURM job per system (25 total).
# Run this ONCE from ~/activeml/scripts/ after uploading the dataset JSONs.
#
# Usage:
#   bash submit_dataset_benchmark.sh
#
# Prerequisites:
#   1. JSON files in ~/activeml/scripts/dataset_jsons/generated300/
#                   ~/activeml/scripts/dataset_jsons/generated_4d5d/
#   2. run_dataset_benchmark.py in ~/activeml/scripts/
#   3. geometry_utils.py   in ~/activeml/scripts/
#   4. All patch files     in ~/activeml/scripts/

SCRIPTS="$HOME/activeml/scripts"
LOG_DIR="/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch"

# All 25 systems
SYSTEMS=(
    # 3d systems
    "CSD_CrCl4_2m_tet_spin4"
    "CSD_MnCl4_2m_tet_spin5"
    "CSD_MnCl6_4m_oct_spin5"
    "CSD_FeCl4_2m_tet_spin4"
    "CSD_FeBr4_2m_tet_spin4"
    "CSD_CoCl4_2m_tet_spin3"
    "CSD_NiCl4_2m_sqpl_spin0"
    "CSD_MnBr4_2m_tet_spin5"
    "CSD_MnF6_4m_oct_spin5"
    "CSD_CrCl4_2m_tet_spin2"
    "CSD_FeCl4_2m_tet_spin0"
    "CSD_FeCl4_dist_tet_spin4"
    "CSD_CoCl6_4m_oct_spin1"
    "CSD_NiCl6_4m_oct_spin2"
    "CSD_MnCl4N2_trans_spin5"
    # 4d systems
    "Mo_Cl6_chg-3_spin3_oct_d2p299"
    "Mo_Cl6_chg-3_spin1_oct_d2p299"
    "Rh_Cl6_chg-3_spin0_oct_d2p32"
    "Ru_Cl6_chg-3_spin1_oct_d2p232"
    "Ru_Cl6_chg-3_spin3_oct_d2p232"
    "Pd_Cl4_chg-2_spin0_sq_pl_d2p3"
    "Mo_Br6_chg-3_spin3_oct_d2p451"
    "Rh_Cl6_chg-3_spin2_oct_d2p32"
    "Rh_Br6_chg-3_spin0_oct_d2p32"
    "Pd_Cl6_chg-2_spin0_oct_d2p3"
)

echo "Submitting ${#SYSTEMS[@]} dataset benchmark jobs..."
echo ""

for SYS in "${SYSTEMS[@]}"; do

# Determine wall time: 4d needs more time
if [[ "$SYS" == Mo_* ]] || [[ "$SYS" == Rh_* ]] || \
   [[ "$SYS" == Ru_* ]] || [[ "$SYS" == Pd_* ]]; then
    WALLTIME="12:00:00"
    MEM="10G"
else
    WALLTIME="08:00:00"
    MEM="8G"
fi

JOBSCRIPT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=ds_${SYS:0:12}
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-qehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${LOG_DIR}/dataset_${SYS}_%j.log
#SBATCH --error=${LOG_DIR}/dataset_${SYS}_%j.err

set -uo pipefail
source ~/.autocas_env.sh || true
set -e

MKL22="/opt/software/pc2/EB-SW/software/imkl/2022.2.1/mkl/2022.2.1/lib/intel64"
GCC11="/opt/software/pc2/EB-SW/software/GCCcore/11.3.0/lib64"
export LD_PRELOAD="\${GCC11}/libgomp.so.1:\${MKL22}/libmkl_gnu_thread.so.2:\${MKL22}/libmkl_core.so.2"
export LD_LIBRARY_PATH="\${MKL22}:\${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export SCRATCH="${LOG_DIR}"

echo "=== Dataset benchmark: ${SYS} ==="
echo "Job: \$SLURM_JOB_ID  Node: \$(hostname)  Start: \$(date)"
echo ""

cd "${SCRIPTS}"
python3 -u run_dataset_benchmark.py ${SYS}

echo ""
echo "Finished: \$(date)"
SLURM
)

    JOBID=$(echo "$JOBSCRIPT" | sbatch --parsable)
    echo "  Submitted ${SYS}: job ${JOBID}"

done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "Results will be in: $SCRIPTS/dataset_benchmark/"
echo ""
echo "Morning check command:"
echo "  for s in \$(ls $SCRIPTS/dataset_benchmark/); do"
echo "    log=\$(ls ${LOG_DIR}/dataset_\${s}_*.log 2>/dev/null | tail -1)"
echo "    echo \"=== \$s ===\" && grep -E 'final_occupation|final_energy|error' \"\$log\" 2>/dev/null | tail -5"
echo "  done"
