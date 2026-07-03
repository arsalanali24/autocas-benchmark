#!/usr/bin/env python3
"""
run_explore_rh_nh3.py

Exploratory autoCAS run for Rh(NH3)6^3+ with reduced DMRG quality.
Purpose: isolate the 3d vs 4d (row) effect by comparing with Co(NH3)6^3+
(same ligand, same d-count, same spin, different row).

Reduced quality settings (M=100, sweeps=3):
  - init_dmrg_bond_dimension: 100   (was 250)
  - init_dmrg_sweeps:          3    (was 5)
  - dmrg_bond_dimension:       100  (was 500)
  - dmrg_sweeps:               3    (was 10)

Expected speedup: ~15-25× vs gold-standard settings.
Expected runtime: 1-3 hours vs 10+ hours for gold standard.

This is EXPLORATORY — lower quality is acceptable here because:
  - We only need to know WHICH orbitals autoCAS selects (size + composition)
  - Absolute energies at M=100 are not publication quality
  - The active space selection is qualitatively stable at M=100 for
    well-separated plateaus

Once the active space is confirmed, a production run at M=250/500 can
follow if needed.

Reference system comparison:
  Co(NH3)6^3+  (done):  3d, d6, S=0 → CAS(4,4)    tiny
  Rh(NH3)6^3+  (this):  4d, d6, S=0 → CAS(??)    isolates row effect
  RhCl6^3-     (done):  4d, d6, S=0 → CAS(22,17)  large

Run:
  source ~/.autocas_env.sh
  sbatch submit_explore_rh_nh3.slurm
"""

import os
import sys
import time
import runpy
import tempfile
import shutil
from pathlib import Path
import yaml

sys.path.insert(0, str(Path.home() / "activeml" / "scripts"))

from patched_pyscf_interface   import PatchedPyscfInterface
from patched_large_spaces      import patch_large_spaces
from patched_qcmaquis_alias    import patch_qcmaquis

import scine_autocas.io.actions.run as run_module
run_module.PyscfInterface = PatchedPyscfInterface
patch_large_spaces()
patch_qcmaquis()

# ── Geometry: Rh(NH3)6^3+ ─────────────────────────────────────────────────
# Rh at origin, 6 NH3 ligands along ±x, ±y, ±z axes
# Rh-N = 2.10 Å (standard for Rh3+ ammine complexes)
# NH3 geometry: N-H = 1.012 Å, H-N-H = 106.7°, umbrella angle = 68.2° from M-N axis
# H positions calculated so NH3 points away from Rh (lone pair → Rh)

XYZ_CONTENT = """\
25
Rh(NH3)6 3+ octahedral Rh-N=2.10 Ang
Rh   0.000   0.000   0.000
N    2.100   0.000   0.000
N   -2.100   0.000   0.000
N    0.000   2.100   0.000
N    0.000  -2.100   0.000
N    0.000   0.000   2.100
N    0.000   0.000  -2.100
H    2.478   0.938   0.000
H    2.478  -0.469   0.813
H    2.478  -0.469  -0.813
H   -2.478   0.938   0.000
H   -2.478  -0.469   0.813
H   -2.478  -0.469  -0.813
H    0.938   2.478   0.000
H   -0.469   2.478   0.813
H   -0.469   2.478  -0.813
H    0.938  -2.478   0.000
H   -0.469  -2.478   0.813
H   -0.469  -2.478  -0.813
H    0.938   0.000   2.478
H   -0.469   0.813   2.478
H   -0.469  -0.813   2.478
H    0.938   0.000  -2.478
H   -0.469   0.813  -2.478
H   -0.469  -0.813  -2.478
"""

# ── autoCAS settings (M=100, sweeps=3 — exploratory quality) ──────────────
YAML_CONTENT = """\
large_cas: true
large_cas_max_orbitals: 30
plateau_values: 10

# REDUCED QUALITY — exploratory run
# M=100 sweeps=3 gives ~15-25x speedup vs gold standard (M=250/500, sweeps=5/10)
# Adequate for active space selection; NOT publication quality for energies
init_dmrg_bond_dimension: 100
init_dmrg_sweeps: 3

fiedler: true

dmrg_bond_dimension: 100
dmrg_sweeps: 3

post_cas_method: nevpt2
basis_set: def2-svp

# Rh: 4d metal, def2-SVP uses 28-electron ECP automatically
# No explicit ecp key needed — PySCF handles this via basis set

dmrg_solver: QCMaquis
uhf: true
double_d_shell: false      # 4d metal: no double d-shell needed
scf_max_cycle: 200
scf_init_guess: minao
scf_symmetry: true         # Oh → D2h gives ~3x speedup

# System: Rh(NH3)6^3+
#   Rh3+: [Kr]4d6 → d6 S=0 (low-spin, strong NH3 field)
#   Charge: +3 (Rh3+ with 6 neutral NH3)
#   Spin: 0 (mol.spin = 0 for singlet)
charge: 3
spin: 0
"""


def main():
    scratch_base = Path(os.environ.get(
        "SCRATCH",
        f"/scratch/hpc-prf-qehpc/{os.environ.get('USER', 'hpcmual')}/autocas_scratch"))

    work_dir = Path.cwd() / "explore_rh_nh3"
    work_dir.mkdir(exist_ok=True)

    # Write geometry and settings
    xyz_file  = work_dir / "rh_nh3.xyz"
    yaml_file = work_dir / "settings.yml"
    xyz_file.write_text(XYZ_CONTENT)
    yaml_file.write_text(YAML_CONTENT)

    print("=" * 60)
    print("Exploratory autoCAS: Rh(NH3)6^3+")
    print("DMRG quality: M=100, sweeps=3 (reduced for exploration)")
    print("=" * 60)
    print(f"Work dir:  {work_dir}")
    print(f"Geometry:  {xyz_file}")
    print(f"Settings:  {yaml_file}")
    print()
    print("Reference comparison:")
    print("  Co(NH3)6^3+  (done):  3d d6 S=0 → CAS(4,4)")
    print("  Rh(NH3)6^3+  (this):  4d d6 S=0 → CAS(??)")
    print("  RhCl6^3-     (done):  4d d6 S=0 → CAS(22,17)")
    print()

    old_dir = os.getcwd()
    os.chdir(work_dir)
    t0 = time.time()

    try:
        sys.argv = [
            "scine_autocas",
            "--xyz",      str(xyz_file),
            "--settings", str(yaml_file),
        ]
        runpy.run_module("scine_autocas", run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n[run] ERROR: {e}")
        raise
    finally:
        os.chdir(old_dir)

    elapsed = time.time() - t0
    print(f"\n[run] completed after {elapsed:.1f}s")
    print(f"[done] check {work_dir}/autocas_project/ for results")
    print()
    print("Key question: is the Rh(NH3)6 active space closer to")
    print("  CAS(4,4)  → NH3 ligand field suppresses correlation")
    print("  CAS(22,17) → 4d covalency dominates regardless of ligand")


if __name__ == "__main__":
    main()
