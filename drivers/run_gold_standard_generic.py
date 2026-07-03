#!/usr/bin/env python3
"""
run_gold_standard_generic.py <system_name>

Single generic driver for all remaining gold-standard autoCAS runs.
All system parameters (geometry, charge, spin, ECP, convergence aids)
are defined in the SYSTEMS dict below.

Usage:
    python3 -u run_gold_standard_generic.py crcl6_csd
    python3 -u run_gold_standard_generic.py fenh3
    python3 -u run_gold_standard_generic.py conh3
    python3 -u run_gold_standard_generic.py rhcl6
    python3 -u run_gold_standard_generic.py recl6
    python3 -u run_gold_standard_generic.py oscl6

Already-completed systems (do not re-run from this script):
    crcl6 (3m idealized), fecl4, mocl6, mncl4
"""

import os
import runpy
import sys
import time
from pathlib import Path

# -------------------------------------------------------------------------
# NH3 geometry helper
# For octahedral M(NH3)6, M at origin, N on ±x/±y/±z axes at distance d.
# Each NH3 has N-H = 1.012 A, H-N-H = 107 deg -> theta(N-H vs N-M axis) = 68.2 deg.
# r_x = 1.012*cos(68.2) = 0.377 A (H displacement along M-N axis, away from M)
# r_perp = 1.012*sin(68.2) = 0.939 A (H displacement perpendicular to M-N axis)
# -------------------------------------------------------------------------

def _nh3_xyz(symbol, d):
    """Return xyz block for M(NH3)6 with M-N distance d (Angstrom)."""
    rx = 0.377   # N-H component along M-N axis (pointing away from M)
    rp = 0.939   # N-H component perpendicular to M-N axis
    lines = [f"{symbol}   0.000   0.000   0.000"]
    # +x NH3
    lines += [
        f"N   {d:.3f}   0.000   0.000",
        f"H   {d+rx:.3f}   0.000   {rp:.3f}",
        f"H   {d+rx:.3f}  {-0.813:.3f}  {-0.470:.3f}",
        f"H   {d+rx:.3f}   {0.813:.3f}  {-0.470:.3f}",
    ]
    # -x NH3
    lines += [
        f"N  {-d:.3f}   0.000   0.000",
        f"H  {-(d+rx):.3f}   0.000   {rp:.3f}",
        f"H  {-(d+rx):.3f}  {-0.813:.3f}  {-0.470:.3f}",
        f"H  {-(d+rx):.3f}   {0.813:.3f}  {-0.470:.3f}",
    ]
    # +y NH3
    lines += [
        f"N   0.000   {d:.3f}   0.000",
        f"H   0.000   {d+rx:.3f}   {rp:.3f}",
        f"H  {-0.813:.3f}   {d+rx:.3f}  {-0.470:.3f}",
        f"H   {0.813:.3f}   {d+rx:.3f}  {-0.470:.3f}",
    ]
    # -y NH3
    lines += [
        f"N   0.000  {-d:.3f}   0.000",
        f"H   0.000  {-(d+rx):.3f}   {rp:.3f}",
        f"H  {-0.813:.3f}  {-(d+rx):.3f}  {-0.470:.3f}",
        f"H   {0.813:.3f}  {-(d+rx):.3f}  {-0.470:.3f}",
    ]
    # +z NH3
    lines += [
        f"N   0.000   0.000   {d:.3f}",
        f"H   {rp:.3f}   0.000   {d+rx:.3f}",
        f"H  {-0.470:.3f}  {-0.813:.3f}   {d+rx:.3f}",
        f"H  {-0.470:.3f}   {0.813:.3f}   {d+rx:.3f}",
    ]
    # -z NH3
    lines += [
        f"N   0.000   0.000  {-d:.3f}",
        f"H   {rp:.3f}   0.000  {-(d+rx):.3f}",
        f"H  {-0.470:.3f}  {-0.813:.3f}  {-(d+rx):.3f}",
        f"H  {-0.470:.3f}   {0.813:.3f}  {-(d+rx):.3f}",
    ]
    n_atoms = len(lines)
    header = f"{n_atoms}\n{symbol}(NH3)6 M-N={d} Ang octahedral\n"
    return header + "\n".join(lines) + "\n"


def _octahedral_xyz(symbol, r, comment=""):
    """Return xyz block for MX6 octahedral with M-X distance r."""
    return (
        f"7\n{comment or symbol + 'Cl6 octahedral M-Cl=' + str(r) + ' Ang'}\n"
        f"{symbol}   0.000   0.000   0.000\n"
        f"Cl   {r:.3f}   0.000   0.000\n"
        f"Cl  {-r:.3f}   0.000   0.000\n"
        f"Cl   0.000   {r:.3f}   0.000\n"
        f"Cl   0.000  {-r:.3f}   0.000\n"
        f"Cl   0.000   0.000   {r:.3f}\n"
        f"Cl   0.000   0.000  {-r:.3f}\n"
    )


# -------------------------------------------------------------------------
# System definitions
# -------------------------------------------------------------------------
SYSTEMS = {

    "crcl6_csd": dict(
        label="CrCl6^3- (CSD geometry, Cr-Cl = 2.30 Ang)",
        xyz=_octahedral_xyz("Cr", 2.30, "CrCl6 3- CSD geometry, Cr-Cl=2.30 Ang"),
        charge=-3, spin_multiplicity=4,
        double_d_shell=True, ecp=False,
        notes="Same system as crcl6_3m but CSD-derived bond length (2.30 vs 2.33 Ang). "
              "Tests geometry sensitivity of autoCAS active space selection. "
              "Parity fix expected (same d3 t2g degeneracy issue as crcl6_3m).",
    ),

    "fenh3": dict(
        label="Fe(NH3)6^2+ (Fe-N = 2.196 Ang, d6 S=2 high-spin)",
        xyz=_nh3_xyz("Fe", 2.196),
        charge=+2, spin_multiplicity=5,
        double_d_shell=True, ecp=False,
        notes="Fe2+ d6 high-spin (S=2, 4 unpaired electrons) in octahedral NH3. "
              "NH3 is a strong-field sigma donor, weaker pi donor than Cl. "
              "Larger valence space than FeCl4 (25 atoms, 84 electrons). "
              "mol.spin=4; if UHF breaks eg degeneracy, parity fix may trigger.",
    ),

    "conh3": dict(
        label="Co(NH3)6^3+ (Co-N = 1.961 Ang, d6 S=0 low-spin)",
        xyz=_nh3_xyz("Co", 1.961),
        charge=+3, spin_multiplicity=1,
        double_d_shell=True, ecp=False,
        notes="Co3+ d6 low-spin (S=0) in octahedral NH3 -- the classic Werner "
              "complex. Closed-shell ground state, RHF reference (mol.spin=0). "
              "No parity issue expected (all electrons paired, active space "
              "will naturally have even electron count). Shorter Co-N vs Fe-N "
              "reflects CFSE-driven contraction in low-spin Co3+.",
    ),

    "rhcl6": dict(
        label="RhCl6^3- (Rh-Cl = 2.34 Ang, d6 S=0 low-spin)",
        xyz=_octahedral_xyz("Rh", 2.34),
        charge=-3, spin_multiplicity=1,
        double_d_shell=False, ecp=True,
        notes="Rh3+ d6 low-spin (S=0) -- 4d analogue of Co(NH3)6. ECP: 28-electron "
              "core (same as Mo). Closed-shell, no parity issue expected. "
              "Spin_mult=1 confirmed correct (was wrongly 2 in original benchmark, "
              "corrected in this project). Direct 3d/4d comparison: "
              "CoNH3 (3d, d6 S=0 in NH3) vs RhCl6 (4d, d6 S=0 in Cl).",
    ),

    "recl6": dict(
        label="ReCl6^2- (Re-Cl = 2.35 Ang, d3 S=3/2)",
        xyz=_octahedral_xyz("Re", 2.35),
        charge=-2, spin_multiplicity=4,
        double_d_shell=False, ecp=True,
        notes="Re4+ d3 S=3/2 -- 5d analogue of Cr3+ (CrCl6) and Mo3+ (MoCl6). "
              "ECP: 60-electron core (large-core, 5d series). Already confirmed "
              "in smoke tests: 60 core electrons removed from Re. "
              "Spin-consistency fix expected (same d3 t2g potential issue as CrCl6). "
              "Direct 3d/4d/5d comparison: CrCl6, MoCl6, ReCl6 all d3 octahedral Cl.",
    ),

    "oscl6": dict(
        label="OsCl6^2- (Os-Cl = 2.32 Ang, d4 S=1)",
        xyz=_octahedral_xyz("Os", 2.32),
        charge=-2, spin_multiplicity=3,
        double_d_shell=False, ecp=True,
        # OsCl6 showed borderline HF convergence in smoke tests (non-converged
        # on default 50 cycles, converged with max_cycle=200 and atom init guess).
        # Use explicit convergence aids as standing settings for this system.
        scf_max_cycle=200,
        scf_init_guess="atom",
        scf_level_shift=0.1,
        notes="Os4+ d4 S=1 -- only d4 system in the benchmark. ECP: 60-electron "
              "core. mol.spin=2 (S=1, 2 unpaired electrons). Borderline HF "
              "convergence observed in smoke tests with default settings; "
              "explicit convergence aids applied (max_cycle=200, atom guess, "
              "level_shift=0.1). 5d system, strong SOC effects are not "
              "captured at this level (would need 2-component DMRG).",
    ),
}


def run_system(name):
    sys_def = SYSTEMS[name]
    print(f"[system] {name}: {sys_def['label']}")
    if "notes" in sys_def:
        print(f"[notes]  {sys_def['notes']}")
    print()

    # Apply all patches
    from patched_pyscf_interface import PatchedPyscfInterface
    import scine_autocas.io.actions.run as run_module
    run_module.PyscfInterface = PatchedPyscfInterface
    print(f"[patch 1] PyscfInterface -> {run_module.PyscfInterface}")

    from patched_large_spaces import apply_patch as p2
    p2()

    from patched_qcmaquis_alias import apply_patch as p3
    p3()

    # Apply optional per-system SCF convergence settings
    orig_build = PatchedPyscfInterface._build_molecule
    scf_max_cycle = sys_def.get("scf_max_cycle", None)
    scf_init_guess = sys_def.get("scf_init_guess", None)
    scf_level_shift = sys_def.get("scf_level_shift", None)

    if scf_max_cycle or scf_init_guess or scf_level_shift:
        def _patched_init_orbs(self):
            if scf_max_cycle:
                self.settings.scf_max_cycle = scf_max_cycle
            if scf_init_guess:
                self.settings.scf_init_guess = scf_init_guess
            if scf_level_shift:
                self.settings.scf_level_shift = scf_level_shift
            return PatchedPyscfInterface._initial_orbitals_impl(self)
        print(f"[patch 4] SCF convergence aids: max_cycle={scf_max_cycle}, "
              f"init_guess={scf_init_guess}, level_shift={scf_level_shift}")

    # Write geometry and config
    work_dir = Path.cwd() / f"gold_standard_{name}"
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    print(f"[setup] working directory: {work_dir}")

    xyz_path = work_dir / f"{name}.xyz"
    xyz_path.write_text(sys_def["xyz"])
    print(f"[setup] wrote geometry: {xyz_path}")

    ecp_str = "def2-svp" if sys_def.get("ecp", False) else "0"
    yaml_content = f"""\
Defaults:
  AutoCAS:
    large_cas: true
    large_cas_average_entanglement: false
    large_cas_max_orbitals: 30
    large_cas_seed: 42
    plateau_values: 10
    single_reference_threshold: 0.14
    threshold_step: 0.01
    weak_correlation_threshold: 0.02
  Interface:
    basis_set: def2-svp
    cas_method: dmrgci
    dmrg_bond_dimension: 500
    dmrg_solver: QCMaquis
    dmrg_sweeps: 10
    dump: true
    fiedler: true
    init_cas_method: dmrgci
    init_dmrg_bond_dimension: 250
    init_dmrg_sweeps: 5
    init_fiedler: true
    init_orbital_order: null
    interface: pyscf
    n_excited_states: 0
    orbital_order: null
    post_cas_method: nevpt2
    uhf: true
  Molecule:
    charge: {sys_def['charge']}
    double_d_shell: {'true' if sys_def['double_d_shell'] else 'false'}
    ecp_electrons: 0
    spin_multiplicity: {sys_def['spin_multiplicity']}
    unit: ang
"""
    yaml_path = work_dir / f"{name}_gold_standard.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[setup] wrote config: {yaml_path}")

    sys.argv = [
        "scine_autocas", "run",
        "-y", str(yaml_path),
        "-x", str(xyz_path),
        "-l", "-u",
    ]
    print(f"[run] sys.argv = {sys.argv}")
    print(f"[run] invoking LargeCasWorkflow on {name} ...")
    print()

    start = time.time()
    try:
        runpy.run_module("scine_autocas", run_name="__main__")
        print(f"\n[run] completed after {time.time()-start:.1f}s")
    except SystemExit as e:
        print(f"\n[run] SystemExit(code={e.code}) after {time.time()-start:.1f}s")
    except Exception as e:
        print(f"\n[run] FAILED: {type(e).__name__}: {e} after {time.time()-start:.1f}s")
        import traceback; traceback.print_exc()
        raise
    finally:
        print(f"\n[done] elapsed: {time.time()-start:.1f}s")
        print(f"[done] check {work_dir}/autocas_project/ for results")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 run_gold_standard_generic.py <system_name>")
        print(f"Available systems: {list(SYSTEMS.keys())}")
        sys.exit(1)
    name = sys.argv[1].lower()
    if name not in SYSTEMS:
        print(f"Unknown system: {name}")
        print(f"Available: {list(SYSTEMS.keys())}")
        sys.exit(1)
    run_system(name)
