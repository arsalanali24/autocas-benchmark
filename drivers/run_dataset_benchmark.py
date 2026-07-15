#!/usr/bin/env python3
"""
run_dataset_benchmark.py  (v3 — fully self-contained, no local file imports)

All geometry logic is INLINED. Does not import geometry_utils.py at all.
Uses the exact same autoCAS CLI invocation as the working benchmark.

Usage:
    source ~/.autocas_env.sh
    python3 -u run_dataset_benchmark.py <system_name>
"""

import sys, os, json, re, math, time, runpy
from pathlib import Path

# ── 23 runnable systems (MnCl4N2=failed, Rh_Br6=missing from zip) ─────────
SYSTEMS_3D = [
    "CSD_CrCl4_2m_tet_spin4",   "CSD_MnCl4_2m_tet_spin5",
    "CSD_MnCl6_4m_oct_spin5",   "CSD_FeCl4_2m_tet_spin4",
    "CSD_FeBr4_2m_tet_spin4",   "CSD_CoCl4_2m_tet_spin3",
    "CSD_NiCl4_2m_sqpl_spin0",  "CSD_MnBr4_2m_tet_spin5",
    "CSD_MnF6_4m_oct_spin5",    "CSD_CrCl4_2m_tet_spin2",
    "CSD_FeCl4_2m_tet_spin0",   "CSD_FeCl4_dist_tet_spin4",
    "CSD_CoCl6_4m_oct_spin1",   "CSD_NiCl6_4m_oct_spin2",
]
SYSTEMS_4D = [
    "Mo_Cl6_chg-3_spin3_oct_d2p299", "Mo_Cl6_chg-3_spin1_oct_d2p299",
    "Rh_Cl6_chg-3_spin0_oct_d2p32",  "Ru_Cl6_chg-3_spin1_oct_d2p232",
    "Ru_Cl6_chg-3_spin3_oct_d2p232", "Pd_Cl4_chg-2_spin0_sq_pl_d2p3",
    "Mo_Br6_chg-3_spin3_oct_d2p451", "Rh_Cl6_chg-3_spin2_oct_d2p32",
    "Pd_Cl6_chg-2_spin0_oct_d2p3",
]
ALL_SYSTEMS = SYSTEMS_3D + SYSTEMS_4D

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path.home() / "activeml" / "scripts"
JSON_DIR    = SCRIPTS_DIR / "dataset_jsons"
RESULTS_DIR = SCRIPTS_DIR / "dataset_benchmark"

# ── Geometry: M-L distances (Angstrom) ────────────────────────────────────
_DIST = {
    ('Cr','Cl','tet'):2.24, ('Cr','Cl','oct'):2.34, ('Cr','F','oct'):1.98,
    ('Cr','Br','tet'):2.39,
    ('Mn','Cl','tet'):2.35, ('Mn','Cl','oct'):2.48, ('Mn','Br','tet'):2.50,
    ('Mn','Br','oct'):2.63, ('Mn','F','tet'):1.90,  ('Mn','F','oct'):1.98,
    ('Fe','Cl','tet'):2.19, ('Fe','Cl','oct'):2.38, ('Fe','Br','tet'):2.35,
    ('Co','Cl','tet'):2.25, ('Co','Cl','oct'):2.44, ('Co','Br','tet'):2.38,
    ('Ni','Cl','sqpl'):2.20,('Ni','Cl','tet'):2.28, ('Ni','Cl','oct'):2.40,
    ('Ni','Br','sqpl'):2.33,
    ('Cu','Cl','sqpl'):2.25,('Zn','Cl','tet'):2.26,
}
_LIG_DIST = {'F':1.95,'Cl':2.30,'Br':2.45,'I':2.65,'O':2.10,'N':2.15}

def _tet(d):
    c = d/math.sqrt(3)
    return [(c,c,c),(-c,-c,c),(-c,c,-c),(c,-c,-c)]

def _oct(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]

def _sqpl(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0)]

_BUILDERS = {'tet':_tet,'oct':_oct,'sqpl':_sqpl,'sq_pl':_sqpl,'jt':_oct}

def _parse_name(name):
    """Parse metal, ligand, n_ligands, geom from system name. Never trust JSON."""
    n = re.sub(r'^CSD_', '', name)
    metal  = (re.match(r'^([A-Z][a-z]?)', n) or re.match(r'.', 'Fe')).group(1)
    m_lig  = re.search(r'_(Cl|Br|F|N|O|I)(\d+)', n) or re.search(r'(Cl|Br|F|N|O|I)(\d+)', n)
    ligand = m_lig.group(1) if m_lig else 'Cl'
    n_lig  = int(m_lig.group(2)) if m_lig else 6
    m_geom = re.search(r'(sq_pl|sqpl|oct|tet|jt)', n, re.IGNORECASE)
    geom   = m_geom.group(1).lower() if m_geom else ('oct' if n_lig==6 else 'tet')
    return metal, ligand, n_lig, geom

def _get_dist(metal, ligand, geom, system):
    if 'dist_ang' in system:
        return float(system['dist_ang'])
    for g in (geom,'tet','oct','sqpl'):
        if (metal,ligand,g) in _DIST:
            return _DIST[(metal,ligand,g)]
    return _LIG_DIST.get(ligand, 2.30)

def build_xyz_string(name, system):
    """Build XYZ file content as a string. Pure Python, no PySCF."""
    metal, ligand, n_lig, geom = _parse_name(name)
    d = _get_dist(metal, ligand, geom, system)
    builder = _BUILDERS.get(geom, _oct)
    lig_positions = builder(d)[:n_lig]

    print(f"[geo] {name}: {geom}, {metal}/{ligand}x{n_lig}, d={d:.3f} Ang")

    lines = [str(1 + n_lig), name,
             f"{metal}  0.000000  0.000000  0.000000"]
    for px, py, pz in lig_positions:
        lines.append(f"{ligand}  {px:.6f}  {py:.6f}  {pz:.6f}")
    return "\n".join(lines) + "\n"

# ── JSON helpers ───────────────────────────────────────────────────────────
def _charge(system):
    for k in ('charge','total_charge','mol_charge'):
        if k in system: return int(system[k])
    m = re.search(r'chg([+-]?\d+)', system.get('name',''))
    if m: return int(m.group(1))
    raise KeyError(f"No charge key. Keys: {list(system.keys())}")

def _spin(system):
    for k in ('spin','n_unpaired'):
        if k in system: return int(system[k])
    for k in ('mult','multiplicity','spin_mult','spin_multiplicity'):
        if k in system: return int(system[k]) - 1
    m = re.search(r'spin(\d+)', system.get('name',''))
    if m: return int(m.group(1))
    raise KeyError(f"No spin key. Keys: {list(system.keys())}")

def find_json(name):
    for sub in ('generated300','generated_4d5d',''):
        p = JSON_DIR / sub / f"{name}.json"
        if p.exists() and p.stat().st_size > 10:
            return p
    raise FileNotFoundError(f"JSON not found for {name}")

# ── Patch loader ───────────────────────────────────────────────────────────
def load_patches():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        from patched_pyscf_interface import PatchedPyscfInterface
        import scine_autocas.io.actions.run as run_module
        run_module.PyscfInterface = PatchedPyscfInterface

        from patched_large_spaces import apply_patch as p2
        p2()
        from patched_qcmaquis_alias import apply_patch as p3
        p3()
        print("[patches] All 3 patches applied")
    except Exception as e:
        print(f"[patches] WARNING: {e}")

# ── autoCAS settings YAML (same format as working benchmark) ───────────────
def make_yaml(name, charge, spin_mult, work_dir, is_4d5d):
    """Generate YAML in the Defaults: format used by run_gold_standard_generic.py"""
    return f"""\
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
    dmrg_bond_dimension: 150
    dmrg_solver: QCMaquis
    dmrg_sweeps: 8
    dump: true
    fiedler: true
    init_cas_method: dmrgci
    init_dmrg_bond_dimension: 150
    init_dmrg_sweeps: 5
    init_fiedler: true
    init_orbital_order: null
    interface: pyscf
    n_excited_states: 0
    orbital_order: null
    post_cas_method: nevpt2
    uhf: true
  Molecule:
    charge: {charge}
    double_d_shell: {'false' if is_4d5d else 'true'}
    ecp_electrons: 0
    spin_multiplicity: {spin_mult}
    unit: ang
"""

# ── Main runner ────────────────────────────────────────────────────────────
def run_system(name, system):
    charge   = _charge(system)
    spin_2s  = _spin(system)
    spin_mult = spin_2s + 1
    is_4d5d  = system.get('metal_row','3d') in ('4d','5d')

    work_dir = RESULTS_DIR / name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write XYZ — pure Python, no PySCF
    xyz_path  = work_dir / f"{name}.xyz"
    xyz_str   = build_xyz_string(name, system)
    xyz_path.write_text(xyz_str)
    print(f"[xyz] Written: {xyz_path}")
    print(f"[xyz] Content preview:\n{xyz_str}")

    # Write YAML
    yaml_path = work_dir / f"{name}.yaml"
    yaml_path.write_text(make_yaml(name, charge, spin_mult, work_dir, is_4d5d))
    print(f"[yaml] Written: {yaml_path}")
    print(f"[info] charge={charge}, spin_mult={spin_mult}, 4d5d={is_4d5d}")

    # Apply patches
    load_patches()

    # Run autoCAS using the SAME CLI as run_gold_standard_generic.py
    scratch = os.environ.get('SCRATCH',
        '/scratch/hpc-prf-qehpc/hpcmual/autocas_scratch')
    old_argv = sys.argv.copy()
    old_cwd  = os.getcwd()
    os.chdir(work_dir)

    sys.argv = [
        "scine_autocas", "run",
        "-y", str(yaml_path),
        "-x", str(xyz_path),
        "-l", "-u",
    ]
    print(f"[run] sys.argv = {sys.argv}")

    t0 = time.time()
    try:
        runpy.run_module("scine_autocas", run_name="__main__")
        elapsed = time.time() - t0
        print(f"[done] {name} completed in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
        return True
    except SystemExit as e:
        elapsed = time.time() - t0
        print(f"[done] SystemExit({e.code}) after {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"[error] {name}: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

# ── Entry point ────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_dataset_benchmark.py <system_name>")
        for s in ALL_SYSTEMS: print(f"  {s}")
        sys.exit(1)

    name = sys.argv[1]
    if name not in ALL_SYSTEMS:
        print(f"ERROR: '{name}' not in system list.")
        sys.exit(1)

    try:
        json_path = find_json(name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)

    system = json.loads(json_path.read_text())
    if 'name' not in system:
        system['name'] = name

    if system.get('status') == 'failed':
        print(f"ERROR: {name} has status=failed in dataset. No geometry available.")
        sys.exit(1)

    print(f"=== Dataset benchmark: {name} ===")
    _, ligand, n_lig, geom = _parse_name(name)
    charge = _charge(system)
    spin   = _spin(system)
    row    = system.get('metal_row','3d')
    print(f"Parsed: {geom}, {ligand}x{n_lig}, charge={charge}, spin(2S)={spin}, row={row}")

    success = run_system(name, system)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
