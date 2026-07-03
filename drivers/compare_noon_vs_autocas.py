#!/usr/bin/env python3
"""
compare_noon_vs_autocas.py

Runs NOON-adaptive CASCI on the 4 completed benchmark systems and compares
against the autoCAS gold-standard reference energies.

Three labeling strategies compared:
  A. Fixed CASSCF(10,10)          — current approach
  B. NOON-adaptive CASCI          — Option 1 proposed alternative
  C. autoCAS CASCI reference      — gold standard (already computed)

NOON-adaptive selection:
  1. Run UHF → spin-averaged natural orbitals
  2. Select all orbitals with NOON in [threshold, 2-threshold]
  3. Cap at max_orbs for tractability
  4. Apply spin-consistency check (Conditions A-D)
  5. Run CASCI

Run on Noctua2:
    source ~/.autocas_env.sh
    python3 -u compare_noon_vs_autocas.py 2>&1 | tee noon_vs_autocas.txt

Output: noon_comparison_output/noon_comparison_report.md
"""

import numpy as np
import time
from pathlib import Path
from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons

OUT = Path("noon_comparison_output")
OUT.mkdir(exist_ok=True)

# autoCAS gold-standard reference (from completed job logs)
AUTOCAS_REF = {
    "FeCl4_1m":  {"ne": 13, "no": 9,  "energy": -3098.455447, "time_h": 3.3},
    "CrCl6_3m":  {"ne": 15, "no": 11, "energy": -3798.961842, "time_h": 8.0},
    "MnCl4_2m":  {"ne": 19, "no": 12, "energy": -2987.317000, "time_h": 6.0},
    "MoCl6_3m":  {"ne": 15, "no": 14, "energy": -2813.770000, "time_h": 3.9},
}

SYSTEMS = {
    "FeCl4_1m": dict(
        atom="Fe 0 0 0; Cl 1.264 1.264 1.264; Cl 1.264 -1.264 -1.264;"
             "Cl -1.264 1.264 -1.264; Cl -1.264 -1.264 1.264",
        charge=-1, spin=5, basis="def2-svp",
        autocas_idx=[30,31,32,33,34,35,36,37,38,39,40,41,46,47,48,49,50],
        autocas_no=9, autocas_ne=13,
    ),
    "CrCl6_3m": dict(
        atom="Cr 0 0 0; Cl 2.33 0 0; Cl -2.33 0 0;"
             "Cl 0 2.33 0; Cl 0 -2.33 0; Cl 0 0 2.33; Cl 0 0 -2.33",
        charge=-3, spin=3, basis="def2-svp",
        autocas_idx=[46,47,53,54,56,57,64,65,66,68,69],
        autocas_no=11, autocas_ne=15,
    ),
    "MnCl4_2m": dict(
        atom="Mn 0 0 0; Cl 1.322 1.322 1.322; Cl 1.322 -1.322 -1.322;"
             "Cl -1.322 1.322 -1.322; Cl -1.322 -1.322 1.322",
        charge=-2, spin=5, basis="def2-svp",
        autocas_idx=None,
        autocas_no=12, autocas_ne=19,
    ),
    "MoCl6_3m": dict(
        atom="Mo 0 0 0; Cl 2.37 0 0; Cl -2.37 0 0;"
             "Cl 0 2.37 0; Cl 0 -2.37 0; Cl 0 0 2.37; Cl 0 0 -2.37",
        charge=-3, spin=3, basis="def2-svp", ecp="def2-svp",
        autocas_idx=None,
        autocas_no=14, autocas_ne=15,
    ),
}


def get_natural_orbitals(mf):
    """UHF spin-averaged NOs. NOONs range 0-2 (full 1-RDM convention)."""
    dm = mf.make_rdm1()
    dm_avg = dm[0] + dm[1]
    s = mf.get_ovlp()
    evals, evecs = np.linalg.eigh(s)
    s_half     = evecs @ np.diag(np.sqrt(evals))       @ evecs.T
    s_half_inv = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    noons_orth, c_orth = np.linalg.eigh(s_half @ dm_avg @ s_half)
    natorbs = s_half_inv @ c_orth
    order = np.argsort(noons_orth)[::-1]
    return natorbs[:, order], noons_orth[order]


def select_noon_adaptive(noons, mol_nelectron, mol_spin,
                         threshold=0.02, max_orbs=24):
    """Select correlated orbitals by NOON threshold with spin-consistency fix."""
    lo, hi = threshold, 2.0 - threshold
    corr_mask = (noons > lo) & (noons < hi)
    idx0 = np.where(corr_mask)[0]
    # Sort by distance from 1.0
    idx0 = idx0[np.argsort(np.abs(noons[idx0] - 1.0))]
    if len(idx0) > max_orbs:
        print(f"  [noon] {len(idx0)} correlated orbitals found, capping at {max_orbs}")
        idx0 = idx0[:max_orbs]
    idx0 = sorted(idx0)

    occ = [int(round(min(max(noons[i], 0), 2))) for i in idx0]
    ne  = int(sum(occ))
    n_singly = sum(1 for o in occ if o == 1)

    # Condition A: ncorelec even
    if (mol_nelectron - ne) % 2 != 0:
        ne += 1
        print(f"  [condA] ne adjusted to {ne}")

    # Condition D: singly-occupied count >= mol_spin
    singly_nearby = sorted(
        [i for i in range(len(noons))
         if abs(noons[i] - 1.0) < 0.3 and i not in idx0],
        key=lambda i: abs(noons[i] - 1.0))

    while n_singly < mol_spin and singly_nearby:
        extra = singly_nearby.pop(0)
        idx0 = sorted(idx0 + [extra])
        occ = [int(round(min(max(noons[i], 0), 2))) for i in idx0]
        ne  = int(sum(occ))
        n_singly = sum(1 for o in occ if o == 1)
        if (mol_nelectron - ne) % 2 != 0:
            ne += 1
        print(f"  [condD] added orb {extra+1} → n_singly={n_singly} CAS({ne},{len(idx0)})")

    return [i+1 for i in idx0], ne, len(idx0)


def run_casci(mf, natorbs, active_1based, ne, label="CASCI"):
    no = len(active_1based)
    if (mf.mol.nelectron - ne) % 2 != 0:
        ne += 1
    mc = mcscf.CASCI(mf, no, ne)
    mo = addons.sort_mo(mc, natorbs, active_1based, base=1)
    t0 = time.time()
    mc.kernel(mo)
    elapsed = time.time() - t0
    dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    cas_noons = np.sort(np.linalg.eigvalsh(dm1))[::-1]
    n_inert = int(np.sum(cas_noons > 1.98) + np.sum(cas_noons < 0.02))
    print(f"  {label}: CAS({ne},{no}) E={mc.e_tot:.6f} Ha  "
          f"t={elapsed:.0f}s  inert={n_inert}/{no}")
    print(f"  CASCI NOONs: {np.round(cas_noons, 3)}")
    return mc.e_tot, cas_noons, mc


def main():
    Ha2kcal = 627.509
    rows = []

    print("=" * 70)
    print("NOON-adaptive CASCI vs autoCAS — label quality comparison")
    print("=" * 70)

    for name, sysdef in SYSTEMS.items():
        ref = AUTOCAS_REF[name]
        print(f"\n{'='*70}")
        print(f"System: {name}")
        print(f"autoCAS gold: CAS({ref['ne']},{ref['no']}) E={ref['energy']:.6f} Ha")

        mol = gto.Mole()
        mol.atom     = sysdef["atom"]
        mol.basis    = sysdef["basis"]
        mol.charge   = sysdef["charge"]
        mol.spin     = sysdef["spin"]
        mol.symmetry = False
        mol.verbose  = 3
        mol.output   = str(OUT / f"{name}_pyscf.log")
        if "ecp" in sysdef:
            mol.ecp = sysdef["ecp"]
        mol.build()

        mf = scf.UHF(mol)
        mf.max_cycle = 200
        mf.init_guess = "minao"
        mf.kernel()
        if not mf.converged:
            mf.init_guess = "atom"; mf.level_shift = 0.1; mf.kernel()
        print(f"UHF E={mf.e_tot:.6f}  <S²>={mf.spin_square()[0]:.3f}")

        natorbs, noons = get_natural_orbitals(mf)
        n_corr = int(np.sum((noons >= 0.02) & (noons <= 1.98)))
        print(f"Correlated orbitals (NOON in [0.02,1.98]): {n_corr}")
        print(f"Their NOONs: {np.round(noons[(noons>=0.02)&(noons<=1.98)], 3)}")

        row = {"system": name,
               "autocas_cas": f"({ref['ne']},{ref['no']})",
               "autocas_e": ref["energy"],
               "n_corr_noon": n_corr}

        # Strategy A: fixed (10,10)
        print(f"\n--- A: fixed (10,10) ---")
        a_idx = sorted(np.argsort(np.abs(noons - 1.0))[:10] + 1)
        ne_a  = int(sum(round(min(max(noons[i-1],0),2)) for i in a_idx))
        if (mol.nelectron - ne_a) % 2 != 0: ne_a += 1
        e_a, _, _ = run_casci(mf, natorbs, a_idx, ne_a, "A (10,10)")
        dE_a = (e_a - ref["energy"]) * Ha2kcal
        row.update({"cas_a": f"({ne_a},10)", "dE_a": dE_a})

        # Strategy B: NOON-adaptive
        print(f"\n--- B: NOON-adaptive (threshold=0.02, max_orbs=24) ---")
        b_idx, ne_b, no_b = select_noon_adaptive(
            noons, mol.nelectron, mol.spin, threshold=0.02, max_orbs=24)
        print(f"Selected {no_b} orbitals: {b_idx}")
        e_b, _, _ = run_casci(mf, natorbs, b_idx, ne_b,
                              f"B NOON({ne_b},{no_b})")
        dE_b = (e_b - ref["energy"]) * Ha2kcal
        row.update({"cas_b": f"({ne_b},{no_b})", "dE_b": dE_b})

        # Strategy C: autoCAS orbitals, CASCI only (no DMRG scan)
        print(f"\n--- C: autoCAS orbitals (CASCI only, no scan) ---")
        if sysdef["autocas_idx"] is not None:
            c_idx = sysdef["autocas_idx"]
        else:
            c_idx = sorted(np.argsort(np.abs(noons-1.0))[:ref["no"]] + 1)
            print(f"  [proxy] using top-{ref['no']} by NOON (exact indices not embedded)")
        e_c, _, _ = run_casci(mf, natorbs, c_idx, ref["ne"],
                              f"C autoCAS({ref['ne']},{ref['no']})")
        dE_c = (e_c - ref["energy"]) * Ha2kcal
        row.update({"cas_c": f"({ref['ne']},{ref['no']})", "dE_c": dE_c})

        # Print summary
        print(f"\n  Results for {name}:")
        print(f"    A (10,10):        {dE_a:+.2f} kcal/mol vs gold")
        print(f"    B NOON-adaptive:  {dE_b:+.2f} kcal/mol vs gold  "
              f"[{ne_b}e, {no_b} orbs]")
        print(f"    C autoCAS CASCI:  {dE_c:+.2f} kcal/mol vs gold")
        print(f"    Improvement A→B:  {dE_a - dE_b:+.2f} kcal/mol")
        print(f"    NOON orbital coverage: {no_b}/{ref['no']} = "
              f"{no_b/ref['no']*100:.0f}% of autoCAS space")
        rows.append(row)

    # Summary report
    write_report(rows)


def write_report(rows):
    Ha2kcal = 627.509
    def flag(de):
        if abs(de) < 1:  return "✓ <1"
        if abs(de) < 5:  return "⚠ 1-5"
        return "✗ >5"

    lines = [
        "# NOON-adaptive CASCI vs autoCAS: Label Quality Report\n",
        "## Energy error vs autoCAS gold standard (kcal/mol)\n",
        "| System | autoCAS space | A: (10,10) | B: NOON-adaptive | "
        "C: autoCAS orbs (CASCI) |",
        "|--------|--------------|-----------|-----------------|"
        "------------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['autocas_cas']} | "
            f"{r['dE_a']:+.1f} {flag(r['dE_a'])} | "
            f"{r['cas_b']} {r['dE_b']:+.1f} {flag(r['dE_b'])} | "
            f"{r['dE_c']:+.1f} {flag(r['dE_c'])} |")

    lines += [
        "\n## What each column tells you\n",
        "- **A vs gold**: how much correlation the fixed (10,10) misses",
        "- **B vs gold**: how much NOON-adaptive CASCI misses (the proposed cheap label)",
        "- **C vs gold**: how much is lost from skipping the DMRG scan while keeping correct orbitals",
        "  - If C is small: scan cost is in *finding* the right orbitals, not computing energy",
        "  - If C is large: the DMRG scan itself contributes to correlation energy\n",
        "## Decision rule for 6100-system labeling\n",
        "| B error | C error | Conclusion |",
        "|---------|---------|------------|",
        "| <1 kcal | <1 kcal | NOON-adaptive is sufficient — use for all 6100 |",
        "| 1-5 kcal| <1 kcal | NOON misses some orbitals; try MP2 NOs or delta correction |",
        "| >5 kcal | <1 kcal | Need better orbital selection upstream of CASCI |",
        "| any     | >5 kcal | Even right orbitals need DMRG — reconsider label type |",
    ]

    report = OUT / "noon_comparison_report.md"
    report.write_text("\n".join(lines))
    print(f"\nReport: {report}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
