#!/bin/bash
# prepare_dataset_jsons.sh
#
# Run on Noctua2 AFTER uploading activeml-dataset-main.zip
# Extracts exactly the 25 needed JSONs and geometry_utils.py
#
# Usage:
#   bash prepare_dataset_jsons.sh activeml-dataset-main.zip

ZIP="${1:-activeml-dataset-main.zip}"
SCRIPTS="$HOME/activeml/scripts"
JSON_DIR="$SCRIPTS/dataset_jsons"

if [ ! -f "$ZIP" ]; then
    echo "ERROR: $ZIP not found"
    echo "Upload the dataset zip first, then run:"
    echo "  bash prepare_dataset_jsons.sh /path/to/activeml-dataset-main.zip"
    exit 1
fi

echo "Extracting dataset JSONs from $ZIP..."
mkdir -p "$JSON_DIR/generated300" "$JSON_DIR/generated_4d5d"

# 3d systems
SYSTEMS_3D=(
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
)

# 4d systems
SYSTEMS_4D=(
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

echo "Extracting 3d JSONs..."
for SYS in "${SYSTEMS_3D[@]}"; do
    unzip -p "$ZIP" "activeml-dataset-main/generated300/${SYS}.json" \
        > "$JSON_DIR/generated300/${SYS}.json" 2>/dev/null && \
        echo "  OK: $SYS" || echo "  MISSING: $SYS"
done

echo "Extracting 4d JSONs..."
for SYS in "${SYSTEMS_4D[@]}"; do
    unzip -p "$ZIP" "activeml-dataset-main/generated_4d5d/${SYS}.json" \
        > "$JSON_DIR/generated_4d5d/${SYS}.json" 2>/dev/null && \
        echo "  OK: $SYS" || echo "  MISSING: $SYS"
done

echo ""
echo "Extracting geometry_utils.py..."
unzip -p "$ZIP" "activeml-dataset-main/dmrg_benchmark/geometry_utils.py" \
    > "$SCRIPTS/geometry_utils.py" && echo "  OK: geometry_utils.py"

echo ""
echo "Verifying extracted files..."
N3D=$(ls "$JSON_DIR/generated300/"*.json 2>/dev/null | wc -l)
N4D=$(ls "$JSON_DIR/generated_4d5d/"*.json 2>/dev/null | wc -l)
echo "  3d systems: $N3D / 15"
echo "  4d systems: $N4D / 10"

if [ -f "$SCRIPTS/geometry_utils.py" ]; then
    echo "  geometry_utils.py: OK"
else
    echo "  geometry_utils.py: MISSING"
fi

echo ""
echo "Next steps:"
echo "  1. Upload run_dataset_benchmark.py to $SCRIPTS/"
echo "  2. Upload submit_dataset_benchmark.sh to $SCRIPTS/"
echo "  3. Run: bash $SCRIPTS/submit_dataset_benchmark.sh"
