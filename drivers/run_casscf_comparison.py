#!/usr/bin/env python3
"""
run_casscf_comparison.py

CASSCF comparison against autoCAS benchmark.
Scientific question: how much does orbital optimization add on top of
the autoCAS-selected space? If CASSCF ≈ CASCI, the autoCAS orbitals
are already near-optimal (QICAS would not help further).
If CASSCF << CASCI, orbital optimization matters.

Systems included (non-5d, confirmed correct autoCAS spaces):
  CoNH3   CAS(4,4)   — trivial CI space
  FeCl4   CAS(13,9)  — small CI space
  CrCl6   CAS(15,11) — small CI space
  MnCl4   CAS(19,12) — small CI space (high spin fills alpha)
  MoCl6   CAS(15,14) — borderline (~6M dets, may take 3-6h)

Skipped:
  RhCl6   CAS(22,17) — 153M dets, needs DMRG solver (block2 not available)
  FeNH3   CAS(?,?)   — autoCAS rerun still needed (Condition D fix)
  OsCl6   CAS(42,28) — 5d, skipped by design
  ReCl6   CAS(9,7)   — 5d, skipped by design

Workflow per system:
  1. UHF (same settings as autoCAS benchmark)
  2. Spin-averaged natural orbitals from UHF 1-RDM
  3. Select autoCAS number of orbitals by NOON proximity to 1.0
  4. Apply spin-consistency check (Conditions A-D)
  5. CASSCF starting from those orbitals
  6. Compare CASSCF energy vs autoCAS CASCI/NEVPT2 reference

Run:
  source ~/.autocas_env.sh
  sbatch submit_casscf_comparison.slurm
"""

import time
import numpy as np
from pathlib import Path
from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons

OUT = Path("casscf_comparison_output")
OUT.mkdir(exist_ok=True)

# ── autoCAS gold-standard CASCI energies (from completed logs) ────────────
AUTOCAS_REF = {
    "CoNH3":  {"ne": 4,  "no": 4,  "e_casci": -1717.195245, "spin": 0},
    "FeCl4":  {"ne": 13, "no": 9,  "e_casci": -3098.455447, "spin": 5},
    "CrCl6":  {"ne": 15, "no": 11, "e_casci": -3798.961842, "spin": 3},
    "MnCl4":  {"ne": 19, "no": 12, "e_casci": -2987.317000, "spin": 5},
    "MoCl6":  {"ne": 15, "no": 14, "e_casci": -2813.770000, "spin": 3},
}

# ── System definitions ────────────────────────────────────────────────────
def nh3_xyz(metal, r_mn):
    """Octahedral M(NH3)6 geometry. r_mn = M-N distance in Ang."""
    # NH3: N-H = 1.012 Ang, H-N-H = 106.7 deg, umbrella angle 68.2 deg
    s, c = 0.927, 0.373   # sin/cos of 68.2 deg
    lines = [f"{metal}  0.000  0.000  0.000"]
    for axis in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        ax, ay, az = axis
        nx, ny, nz = ax*r_mn, ay*r_mn, az*r_mn
        lines.append(f"N  {nx:.3f}  {ny:.3f}  {nz:.3f}")
        # 3 H atoms around N, C3 axis along M-N direction
        nh = 1.012
        if ax != 0:
            hs = [(nh*(ax*c), nh*s, 0),
                  (nh*(ax*c), -nh*s*0.5, nh*s*0.866),
                  (nh*(ax*c), -nh*s*0.5, -nh*s*0.866)]
        elif ay != 0:
            hs = [(nh*s, nh*(ay*c), 0),
                  (-nh*s*0.5, nh*(ay*c), nh*s*0.866),
                  (-nh*s*0.5, nh*(ay*c), -nh*s*0.866)]
        else:
            hs = [(nh*s, 0, nh*(az*c)),
                  (-nh*s*0.5, nh*s*0.866, nh*(az*c)),
                  (-nh*s*0.5, -nh*s*0.866, nh*(az*c))]
        for hx, hy, hz in hs:
            lines.append(f"H  {nx+hx:.3f}  {ny+hy:.3f}  {nz+hz:.3f}")
    return "\n".join(lines)

def oct_xyz(metal, r):
    """Octahedral MCl6 geometry."""
    lines = [f"{metal}  0.000  0.000  0.000"]
    for ax, ay, az in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        lines.append(f"Cl  {ax*r:.3f}  {ay*r:.3f}  {az*r:.3f}")
    return "\n".join(lines)

def tet_xyz(metal, r):
    """Tetrahedral MCl4 geometry."""
    lines = [f"{metal}  0.000  0.000  0.000"]
    for sx, sy, sz in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]:
        lines.append(f"Cl  {sx*r:.3f}  {sy*r:.3f}  {sz*r:.3f}")
    return "\n".join(lines)

SYSTEMS = {
    "CoNH3": dict(
        atom=nh3_xyz("Co", 1.961),
        charge=3, spin=0, basis="def2-svp",
        note="Co3+ d6 S=0 low-spin, CAS(4,4) — trivial CI"),
    "FeCl4": dict(
        atom=tet_xyz("Fe", 1.264),
        charge=-1, spin=5, basis="def2-svp",
        note="Fe3+ d5 S=5/2 tetrahedral, CAS(13,9)"),
    "CrCl6": dict(
        atom=oct_xyz("Cr", 2.33),
        charge=-3, spin=3, basis="def2-svp",
        note="Cr3+ d3 S=3/2 octahedral, CAS(15,11)"),
    "MnCl4": dict(
        atom=tet_xyz("Mn", 1.322),
        charge=-2, spin=5, basis="def2-svp",
        note="Mn2+ d5 S=5/2 tetrahedral, CAS(19,12)"),
    "MoCl6": dict(
        atom=oct_xyz("Mo", 2.37),
        charge=-3, spin=3, basis="def2-svp", ecp="def2-svp",
        note="Mo3+ d3 S=3/2 octahedral, CAS(15,14) — largest CI space"),
}


# ── Utilities (same as validate_casscf1010_labels.py) ────────────────────

def get_natural_orbitals(mf):
    """Spin-averaged UHF natural orbitals. NOONs range 0-2."""
    dm = mf.make_rdm1()
    dm_avg = dm[0] + dm[1]
    s = mf.get_ovlp()
    evals, evecs = np.linalg.eigh(s)
    s_half     = evecs @ np.diag(np.sqrt(evals))       @ evecs.T
    s_half_inv = evecs @ np.diag(1.0/np.sqrt(evals))   @ evecs.T
    noons_orth, c_orth = np.linalg.eigh(s_half @ dm_avg @ s_half)
    natorbs = s_half_inv @ c_orth
    order = np.argsort(noons_orth)[::-1]
    return natorbs[:, order], noons_orth[order]


def select_and_fix(noons, mol_nelectron, mol_spin, n_orbs):
    """Select n_orbs by NOON proximity to 1.0 with spin-consistency."""
    dist = np.abs(noons - 1.0)
    idx0 = sorted(np.argsort(dist)[:n_orbs])
    occ  = [int(round(min(max(noons[i], 0), 2))) for i in idx0]
    ne   = int(sum(occ))
    # Condition A: ncorelec even
    if (mol_nelectron - ne) % 2 != 0:
        ne += 1
    # Condition D: n_singly >= mol_spin
    n_singly = sum(1 for o in occ if o == 1)
    singly_extra = sorted(
        [i for i in range(len(noons))
         if abs(noons[i]-1.0) < 0.3 and i not in idx0],
        key=lambda i: abs(noons[i]-1.0))
    while n_singly < mol_spin and singly_extra:
        extra = singly_extra.pop(0)
        idx0 = sorted(idx0 + [extra])
        occ  = [int(round(min(max(noons[i],0),2))) for i in idx0]
        ne   = int(sum(occ))
        if (mol_nelectron - ne) % 2 != 0:
            ne += 1
        n_singly = sum(1 for o in occ if o == 1)
    active_1based = [i+1 for i in idx0]
    return active_1based, ne, len(idx0)


def run_uhf(mol):
    mf = scf.UHF(mol)
    mf.max_cycle = 200
    mf.init_guess = "minao"
    mf.kernel()
    if not mf.converged:
        mf.init_guess = "atom"
        mf.level_shift = 0.1
        mf.kernel()
    return mf


def run_casscf(mf, natorbs, active_1based, ne, label="CASSCF"):
    """Run CASSCF with natural orbital starting guess."""
    no = len(active_1based)
    if (mf.mol.nelectron - ne) % 2 != 0:
        ne += 1
    # CASSCF with (na, nb) tuple to be explicit about spin
    mol_spin = mf.mol.spin
    na = (ne + mol_spin) // 2
    nb = (ne - mol_spin) // 2
    mc = mcscf.CASSCF(mf, no, (na, nb))
    mc.max_cycle_macro = 50
    mc.max_cycle_micro = 10
    mc.conv_tol = 1e-8
    mc.verbose = 4
    mc.output = str(OUT / f"{label.replace(' ','_')}.log")
    mo = addons.sort_mo(mc, natorbs, active_1based, base=1)
    t0 = time.time()
    mc.kernel(mo)
    elapsed = time.time() - t0
    converged = "CONVERGED" if mc.converged else "NOT CONVERGED"
    print(f"  {label} CAS({ne},{no}): E={mc.e_tot:.6f} Ha  "
          f"t={elapsed:.0f}s  {converged}")
    return mc.e_tot, mc.converged


# ── Main loop ─────────────────────────────────────────────────────────────

def main():
    Ha2kcal = 627.509
    results = []

    print("=" * 65)
    print("CASSCF comparison: autoCAS space vs CASCI reference")
    print("Scientific question: does orbital optimization matter")
    print("on top of autoCAS-selected orbitals?")
    print("=" * 65)

    for name, sysdef in SYSTEMS.items():
        ref = AUTOCAS_REF[name]
        print(f"\n{'='*65}")
        print(f"System: {name}  ({sysdef['note']})")
        print(f"autoCAS reference: CAS({ref['ne']},{ref['no']})  "
              f"E_CASCI = {ref['e_casci']:.6f} Ha")

        # Build molecule
        mol = gto.Mole()
        mol.atom     = sysdef["atom"]
        mol.basis    = sysdef["basis"]
        mol.charge   = sysdef["charge"]
        mol.spin     = sysdef["spin"]
        mol.symmetry = False
        mol.verbose  = 3
        mol.output   = str(OUT / f"{name}_uhf.log")
        if "ecp" in sysdef:
            mol.ecp = sysdef["ecp"]
        mol.build()

        # UHF
        mf = run_uhf(mol)
        print(f"UHF: E={mf.e_tot:.6f}  <S²>={mf.spin_square()[0]:.3f}")

        # Natural orbitals
        natorbs, noons = get_natural_orbitals(mf)

        # Select autoCAS-sized active space from NOONs
        active_idx, ne, no = select_and_fix(
            noons, mol.nelectron, mol.spin, ref["no"])
        print(f"NOON-selected active space: CAS({ne},{no})")
        print(f"Active orbital indices (1-based): {active_idx}")
        print(f"Selected NOONs: "
              f"{np.round(noons[[i-1 for i in active_idx]], 3)}")

        # CASSCF
        e_casscf, conv = run_casscf(
            mf, natorbs, active_idx, ne,
            label=f"{name}_CASSCF")

        # Comparison
        dE = (e_casscf - ref["e_casci"]) * Ha2kcal
        dE_mha = (e_casscf - ref["e_casci"]) * 1000  # mHa
        print(f"\n  Result for {name}:")
        print(f"    autoCAS CASCI:  {ref['e_casci']:.6f} Ha")
        print(f"    CASSCF:         {e_casscf:.6f} Ha")
        print(f"    ΔE (CASSCF-CASCI): {dE:+.2f} kcal/mol  "
              f"({dE_mha:+.1f} mHa)")
        if abs(dE) < 1.0:
            verdict = "CASCI ≈ CASSCF: autoCAS orbitals near-optimal"
        elif abs(dE) < 5.0:
            verdict = "moderate gap: orbital optimization helps"
        else:
            verdict = "large gap: orbital optimization important"
        print(f"    Verdict: {verdict}")

        results.append({
            "system": name,
            "cas": f"({ne},{no})",
            "e_casci": ref["e_casci"],
            "e_casscf": e_casscf,
            "dE_kcal": dE,
            "converged": conv,
        })

    # Summary report
    print(f"\n{'='*65}")
    print("SUMMARY — CASSCF vs autoCAS CASCI")
    print(f"{'='*65}")
    print(f"{'System':8} {'Space':10} {'CASCI (Ha)':18} "
          f"{'CASSCF (Ha)':18} {'ΔE (kcal/mol)':15} {'Conv':5}")
    print("-" * 78)
    for r in results:
        print(f"{r['system']:8} {r['cas']:10} {r['e_casci']:18.6f} "
              f"{r['e_casscf']:18.6f} {r['dE_kcal']:+15.2f} "
              f"{'✓' if r['converged'] else '✗':5}")

    # Write markdown
    md = ["# CASSCF vs autoCAS CASCI Comparison\n",
          "Does orbital optimization (CASSCF) improve on the autoCAS-selected space?\n",
          "| System | Space | CASCI (Ha) | CASSCF (Ha) | ΔE (kcal/mol) | Converged |",
          "|--------|-------|-----------|------------|---------------|-----------|"]
    for r in results:
        flag = ("✓ near-opt" if abs(r['dE_kcal'])<1
                else "⚠ moderate" if abs(r['dE_kcal'])<5
                else "✗ large gap")
        md.append(
            f"| {r['system']} | {r['cas']} | {r['e_casci']:.6f} | "
            f"{r['e_casscf']:.6f} | {r['dE_kcal']:+.2f} {flag} | "
            f"{'yes' if r['converged'] else 'no'} |")
    md += ["\n## Interpretation",
           "- ΔE < 1 kcal/mol: autoCAS orbitals already near-optimal; "
           "QICAS would not help significantly",
           "- ΔE 1-5 kcal/mol: orbital optimization adds moderate improvement",
           "- ΔE > 5 kcal/mol: wrong orbitals selected; orbital optimization essential"]
    (OUT/"casscf_comparison_report.md").write_text("\n".join(md))
    print(f"\nReport: {OUT/'casscf_comparison_report.md'}")


if __name__ == "__main__":
    main()
