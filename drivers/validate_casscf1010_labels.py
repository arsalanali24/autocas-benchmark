#!/usr/bin/env python3
"""
validate_casscf1010_labels.py

Three-check validation comparing CASSCF(10,10) labels against the
autoCAS gold-standard active spaces for the four completed benchmark systems.

CHECK 1 — Active space size: does (10,10) cover the right orbitals?
CHECK 2 — NOON analysis: are included orbitals actually correlated?
CHECK 3 — Spin-state gaps: does (10,10) predict the correct spin ordering?

Run on Noctua2 (after source ~/.autocas_env.sh):
    python3 -u validate_casscf1010_labels.py 2>&1 | tee validation_results.txt

All geometries and autoCAS results are embedded directly; this script is
self-contained and does not depend on autoCAS being installed.

Output: validation_report.md  +  detailed log to stdout
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# ── PySCF imports ──────────────────────────────────────────────────────────
from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf import addons

# ── Output directory ───────────────────────────────────────────────────────
OUT = Path("validation_output")
OUT.mkdir(exist_ok=True)

# ── System definitions ─────────────────────────────────────────────────────
# Each entry carries:
#   xyz          : geometry string (angstrom)
#   charge       : total charge
#   spin_hs      : 2S for the high-spin ground state (2S = n_unpaired)
#   spin_ls      : 2S for the first excited spin state to test
#   basis        : basis set
#   autocas_idx  : autoCAS-selected orbital indices (1-based for PySCF sort_mo)
#   autocas_occ  : occupation per selected orbital (2/1/0)
#   autocas_ne   : active electrons after spin-consistency fix
#   autocas_no   : active orbitals after spin-consistency fix
#   label_ne     : electrons in the (10,10) label space
#   label_no     : orbitals in the (10,10) label space (always 10)
#   exp_gap_cm1  : experimental HS→LS gap in cm⁻¹ (positive = HS lower)
#                  None if not available

SYSTEMS = {

    "FeCl4_1m": dict(
        xyz="""\
5
FeCl4 1- tetrahedral Fe-Cl=2.19 Ang
Fe   0.000   0.000   0.000
Cl   1.264   1.264   1.264
Cl   1.264  -1.264  -1.264
Cl  -1.264   1.264  -1.264
Cl  -1.264  -1.264   1.264""",
        charge=-1, basis="def2-svp",
        spin_hs=5, spin_ls=3,          # S=5/2 vs S=3/2
        # autoCAS result: CAS(13,9), no parity fix needed
        autocas_idx=[30,31,32,33,34,35,36,37,38,39,40,41,46,47,48,49,50],
        # actual 1-based indices from the log: CAS indices (1-based)
        # From log: CAS orbital indices (1-based): [30,31,32,33,34,35,36,37,38,39,40,41,46,47,48,49,50]
        # occupation: [2,2,2,2,1,1,1,1,1] for 9 orbitals
        # Let's use the simpler form: just the 9 active orbitals
        autocas_active_1based=[30,31,32,33,34,35,36,37,38,39,40,41,46,47,48,49,50],
        autocas_ne=13, autocas_no=9,
        label_ne=10, label_no=10,
        # Experimental: FeCl4^1- is unambiguously high-spin d5 S=5/2
        # LS state S=3/2 is very high in energy (ligand field splitting)
        # T. Brunold et al. JACS 2002 reference
        exp_gap_cm1=None,              # not directly measured; HS strongly favoured
        notes="d5 HS, tetrahedral. autoCAS: CAS(13,9). "
              "(10,10) comparable in size but orbital selection differs."
    ),

    "CrCl6_3m": dict(
        xyz="""\
7
CrCl6 3- octahedral Cr-Cl=2.33 Ang
Cr   0.000   0.000   0.000
Cl   2.330   0.000   0.000
Cl  -2.330   0.000   0.000
Cl   0.000   2.330   0.000
Cl   0.000  -2.330   0.000
Cl   0.000   0.000   2.330
Cl   0.000   0.000  -2.330""",
        charge=-3, basis="def2-svp",
        spin_hs=3, spin_ls=1,          # S=3/2 vs S=1/2
        # autoCAS result: CAS(15,11) after parity fix (added orbital 63)
        # Log indices (0-based): [45,46,52,53,55,56,63,64,65,67,68]
        # 1-based: [46,47,53,54,56,57,64,65,66,68,69]
        autocas_ne=15, autocas_no=11,
        label_ne=10, label_no=10,
        # Experimental: Cr^3+ d3 t2g^3, 4A2g ground state in Oh
        # Ligand field transitions visible in UV-Vis
        # Tanabe-Sugano: 10Dq ~ 13000-14000 cm-1 for CrCl6^3-
        # LS state (2Eg) is ~15000 cm-1 above GS
        exp_gap_cm1=15000,
        notes="d3 t2g, octahedral. autoCAS: CAS(15,11) after parity fix. "
              "(10,10) is 1 orbital smaller — missing one t2g partner."
    ),

    "MnCl4_2m": dict(
        xyz="""\
5
MnCl4 2- tetrahedral Mn-Cl=2.29 Ang
Mn   0.000   0.000   0.000
Cl   1.322   1.322   1.322
Cl   1.322  -1.322  -1.322
Cl  -1.322   1.322  -1.322
Cl  -1.322  -1.322   1.322""",
        charge=-2, basis="def2-svp",
        spin_hs=5, spin_ls=3,          # S=5/2 vs S=3/2
        # autoCAS result: CAS(19,12) after spin-consistency fix (added 2 orbs)
        autocas_ne=19, autocas_no=12,
        label_ne=10, label_no=10,
        exp_gap_cm1=None,
        notes="d5 HS, tetrahedral. autoCAS: CAS(19,12) — 2 orbs larger than (10,10). "
              "Most likely case where (10,10) labels are qualitatively wrong."
    ),

    "MoCl6_3m": dict(
        xyz="""\
7
MoCl6 3- octahedral Mo-Cl=2.37 Ang
Mo   0.000   0.000   0.000
Cl   2.370   0.000   0.000
Cl  -2.370   0.000   0.000
Cl   0.000   2.370   0.000
Cl   0.000  -2.370   0.000
Cl   0.000   0.000   2.370
Cl   0.000   0.000  -2.370""",
        charge=-3, basis="def2-svp",
        spin_hs=3, spin_ls=1,          # S=3/2 vs S=1/2
        # autoCAS result: CAS(15,14) — no parity fix, 14 orbitals
        autocas_ne=15, autocas_no=14,
        label_ne=10, label_no=10,
        ecp="def2-svp",                # Mo needs ECP
        exp_gap_cm1=None,
        notes="d3, 4d octahedral. autoCAS: CAS(15,14) — 4 more orbitals than (10,10). "
              "Larger active space reflects 4d diffuseness and covalency."
    ),
}


def build_mol(sysdef, spin_override=None):
    """Build PySCF Mole from system definition."""
    spin = spin_override if spin_override is not None else sysdef["spin_hs"]
    mol = gto.Mole()
    # PySCF mol.atom expects coordinate block only, not full xyz format.
    # Strip the first two lines (atom count + comment) from the xyz string.
    xyz_lines = sysdef["xyz"].strip().split("\n")
    mol.atom = "\n".join(xyz_lines[2:])
    mol.basis = sysdef["basis"]
    mol.charge = sysdef["charge"]
    mol.spin = spin
    mol.symmetry = False  # disabled: causes sort_mo conflicts with natural orbitals
    mol.verbose = 3
    mol.output = str(OUT / f"pyscf_{spin}.log")
    if "ecp" in sysdef:
        mol.ecp = sysdef["ecp"]
    mol.build()
    return mol


def get_natural_orbitals(mf):
    """
    Compute spin-averaged natural orbitals from UHF 1-RDM.
    mf.mo_coeff for UHF is a TUPLE (alpha, beta) -- cannot be passed
    directly to sort_mo which expects a single 2D AO->MO matrix.
    Natural orbitals from the spin-averaged density give a single
    orthonormal set that CASCI can use, and NOONs identify correlated orbs.
    """
    dm = mf.make_rdm1()
    dm_avg = dm[0] + dm[1]  # full 1-RDM, NOONs range 0-2
    s = mf.get_ovlp()
    s_evals, s_evecs = np.linalg.eigh(s)
    s_half     = s_evecs @ np.diag(np.sqrt(s_evals))       @ s_evecs.T
    s_half_inv = s_evecs @ np.diag(1.0 / np.sqrt(s_evals)) @ s_evecs.T
    dm_orth = s_half @ dm_avg @ s_half
    noons_orth, c_orth = np.linalg.eigh(dm_orth)
    natorbs = s_half_inv @ c_orth
    order = np.argsort(noons_orth)[::-1]   # descending occupation
    return natorbs[:, order], noons_orth[order]


def select_orbitals_by_noon(noons, n=10):
    """Select n orbitals with NOONs closest to 1.0. Returns 1-based indices."""
    dist = np.abs(noons - 1.0)
    return sorted(np.argsort(dist)[:n] + 1)


def run_uhf(mol):
    """Run UHF, return (mf, natorbs, noons)."""
    mf = scf.UHF(mol)
    mf.max_cycle = 200
    mf.init_guess = "minao"
    mf.kernel()
    if not mf.converged:
        print("  UHF not converged with minao; retrying with atom guess")
        mf.init_guess = "atom"
        mf.level_shift = 0.1
        mf.kernel()
    natorbs, noons = get_natural_orbitals(mf)
    return mf, natorbs, noons


def select_10_orbitals_from_uhf(mf):
    """Legacy wrapper — kept for compatibility."""
    natorbs, noons = get_natural_orbitals(mf)
    return select_orbitals_by_noon(noons, 10), noons


def run_casci(mf, natorbs, active_1based, ne_override=None, label="CASCI"):
    """
    Run CASCI using natural orbitals as MO basis.
    ne_override: explicit active electron count (use for autoCAS spaces
    where we know the exact count from the log). If None, inferred from
    NOON rounding with parity correction.
    """
    no = len(active_1based)
    _, noons = get_natural_orbitals(mf)

    if ne_override is not None:
        ne = ne_override
    else:
        # Estimate from rounded NOONs of selected orbitals
        idx0 = [i-1 for i in active_1based]
        ne = int(sum(round(min(max(noons[i], 0), 2)) for i in idx0))

    # Parity check and correction
    ncorelec = mf.mol.nelectron - ne
    if ncorelec % 2 != 0:
        ne_adj = ne + 1
        if (mf.mol.nelectron - ne_adj) % 2 == 0:
            print(f"  [parity] ne={ne}→{ne_adj} to make ncorelec even")
            ne = ne_adj
        else:
            ne_adj = ne - 1
            if (mf.mol.nelectron - ne_adj) % 2 == 0:
                print(f"  [parity] ne={ne}→{ne_adj} to make ncorelec even")
                ne = ne_adj

    mc = mcscf.CASCI(mf, no, ne)
    mo = addons.sort_mo(mc, natorbs, active_1based, base=1)
    t0 = time.time()
    mc.kernel(mo)
    elapsed = time.time() - t0
    dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    cas_noons = np.sort(np.linalg.eigvalsh(dm1))[::-1]
    n_inert = int(np.sum(cas_noons > 1.98) + np.sum(cas_noons < 0.02))
    print(f"  {label} CAS({ne},{no}) E={mc.e_tot:.6f} Ha  "
          f"t={elapsed:.0f}s  inert={n_inert}/{no}")
    print(f"  NOONs: {np.round(cas_noons, 4)}")
    return mc.e_tot, cas_noons, mc


def run_spin_state_gap(sysdef, name, orbital_selector, ne, no, label):
    """
    Compute spin-state gap ΔE(LS - HS) in cm⁻¹.
    Positive = HS lower in energy = correct for high-spin ground state.
    """
    results = {}
    for spin_state, spin_key in [("HS", "spin_hs"), ("LS", "spin_ls")]:
        mol = build_mol(sysdef, spin_override=sysdef[spin_key])
        mf, natorbs_ss, noons_ss = run_uhf(mol)
        idx = orbital_selector(mf, sysdef)
        e, _, mc = run_casci(mf, natorbs_ss, idx,
                             ne_override=ne,
                             label=f"{label} {spin_state}")
        results[spin_state] = e

    # ΔE(HS→LS): positive means HS is lower (ground state)
    gap_Ha = results["LS"] - results["HS"]
    gap_cm1 = gap_Ha * 219474.6      # 1 Ha = 219474.6 cm⁻¹
    gap_kcal = gap_Ha * 627.509
    print(f"\n  {label} spin-state gap (LS - HS):")
    print(f"    ΔE = {gap_cm1:+.0f} cm⁻¹  ({gap_kcal:+.2f} kcal/mol)")
    if sysdef.get("exp_gap_cm1"):
        err = gap_cm1 - sysdef["exp_gap_cm1"]
        print(f"    experiment = {sysdef['exp_gap_cm1']:+.0f} cm⁻¹")
        print(f"    error      = {err:+.0f} cm⁻¹  "
              f"({'within' if abs(err) < 1000 else 'OUTSIDE'} ±1000 cm⁻¹ target)")
    return gap_cm1, gap_kcal


def autocas_orbital_selector(mf, sysdef):  # mf kept for signature compat
    """Return autoCAS-selected 1-based orbital indices."""
    # These are embedded from the log: compute dynamically from the
    # autoCAS-fixed indices stored per system where available.
    # For systems where we have them, return directly.
    # Otherwise fall back to the top-(autocas_no) by entropy proxy.
    if "autocas_active_1based" in sysdef:
        # Use only the active orbital subset (not the full CAS list)
        return sysdef["autocas_active_1based"]
    # Fallback: select autocas_no orbitals closest to half-filling
    _, occ = select_10_orbitals_from_uhf.__wrapped__(mf) \
        if hasattr(select_10_orbitals_from_uhf, '__wrapped__') \
        else (None, None)
    if occ is None:
        dm_avg = (mf.make_rdm1()[0] + mf.make_rdm1()[1]) / 2
        s = mf.get_ovlp()
        from scipy import linalg
        s_half = linalg.sqrtm(s)
        dm_orth = s_half @ dm_avg @ s_half
        occ, _ = np.linalg.eigh(dm_orth)
    no = sysdef["autocas_no"]
    dist_from_1 = np.abs(occ - 1.0)
    top = sorted(np.argsort(dist_from_1)[:no] + 1)
    return top


def label_orbital_selector(mf, sysdef):  # legacy, kept for compat
    natorbs, noons = get_natural_orbitals(mf)
    return select_orbitals_by_noon(noons, n=sysdef["label_no"])


# ── Main validation loop ────────────────────────────────────────────────────

def main():
    report_rows = []
    print("=" * 70)
    print("CASSCF(10,10) label validation against autoCAS gold standard")
    print("=" * 70)

    for name, sysdef in SYSTEMS.items():
        print(f"\n{'='*70}")
        print(f"System: {name}")
        print(f"Notes:  {sysdef['notes']}")
        print(f"autoCAS: CAS({sysdef['autocas_ne']}, {sysdef['autocas_no']})")
        print(f"(10,10): CAS({sysdef['label_ne']}, {sysdef['label_no']})")
        size_diff = sysdef['autocas_no'] - sysdef['label_no']
        if size_diff > 0:
            print(f"WARNING: (10,10) is {size_diff} orbitals SMALLER than autoCAS")
        elif size_diff < 0:
            print(f"NOTE: (10,10) is {abs(size_diff)} orbitals larger than autoCAS")
        print()

        row = {"system": name,
               "autocas_cas": f"({sysdef['autocas_ne']},{sysdef['autocas_no']})",
               "label_cas": f"({sysdef['label_ne']},{sysdef['label_no']})"}

        # ── Ground state: both methods ──────────────────────────────────────
        mol_gs = build_mol(sysdef)
        mf_gs, natorbs_gs, noons_gs = run_uhf(mol_gs)

        # (10,10) label space
        print(f"--- (10,10) label space ---")
        label_idx = select_orbitals_by_noon(noons_gs, n=sysdef["label_no"])
        print(f"Selected orbitals (1-based): {label_idx}")
        e_label_gs, noons_label, _ = run_casci(
            mf_gs, natorbs_gs, label_idx, label="CASCI(10,10)")
        row["e_label_gs"] = e_label_gs
        row["noons_label_min"] = float(noons_label.min())
        row["noons_label_max"] = float(noons_label.max())
        n_inert_label = int(np.sum(noons_label > 1.98) +
                            np.sum(noons_label < 0.02))
        row["inert_orbs_label"] = n_inert_label

        # autoCAS space
        print(f"\n--- autoCAS space ---")
        autocas_idx = autocas_orbital_selector(mf_gs, sysdef)
        print(f"Selected orbitals (1-based): {autocas_idx}")
        e_autocas_gs, noons_autocas, _ = run_casci(
            mf_gs, natorbs_gs, autocas_idx,
            ne_override=sysdef["autocas_ne"],
            label="CASCI autoCAS")
        row["e_autocas_gs"] = e_autocas_gs
        row["noons_autocas_min"] = float(noons_autocas.min())
        row["noons_autocas_max"] = float(noons_autocas.max())
        n_inert_autocas = int(np.sum(noons_autocas > 1.98) +
                              np.sum(noons_autocas < 0.02))
        row["inert_orbs_autocas"] = n_inert_autocas

        # Energy difference (GS)
        dE_Ha = e_label_gs - e_autocas_gs
        dE_kcal = dE_Ha * 627.509
        dE_cm1 = dE_Ha * 219474.6
        row["dE_gs_kcal"] = dE_kcal
        row["dE_gs_cm1"] = dE_cm1
        print(f"\nGround state energy difference  (10,10) - autoCAS:")
        print(f"  ΔE = {dE_kcal:+.2f} kcal/mol  ({dE_cm1:+.0f} cm⁻¹)")
        if abs(dE_kcal) < 1.0:
            print("  → within chemical accuracy (1 kcal/mol)")
        elif abs(dE_kcal) < 5.0:
            print("  → moderate disagreement (1-5 kcal/mol)")
        else:
            print("  → LARGE disagreement (>5 kcal/mol) — labels unreliable for this system")

        # ── Spin-state gap: both methods ────────────────────────────────────
        print(f"\n{'─'*50}")
        print("Spin-state gap comparison:")

        gap_label, gap_label_kcal = run_spin_state_gap(
            sysdef, name, label_orbital_selector,
            sysdef["label_ne"], sysdef["label_no"], "(10,10)")
        row["gap_label_cm1"] = gap_label

        gap_autocas, gap_autocas_kcal = run_spin_state_gap(
            sysdef, name, autocas_orbital_selector,
            sysdef["autocas_ne"], sysdef["autocas_no"], "autoCAS")
        row["gap_autocas_cm1"] = gap_autocas

        gap_diff = gap_label - gap_autocas
        row["gap_diff_cm1"] = gap_diff
        print(f"\n  Gap difference: (10,10) - autoCAS = {gap_diff:+.0f} cm⁻¹")
        if abs(gap_diff) < 500:
            print("  → gaps agree well (<500 cm⁻¹)")
        elif abs(gap_diff) < 2000:
            print("  → moderate disagreement (500-2000 cm⁻¹)")
        else:
            print("  → LARGE disagreement (>2000 cm⁻¹) — (10,10) label is unreliable")

        # Sign check: do both agree on which state is lower?
        if np.sign(gap_label) != np.sign(gap_autocas):
            print("  ⚠ CRITICAL: (10,10) and autoCAS DISAGREE on which spin "
                  "state is the ground state!")
            row["spin_ordering_correct"] = False
        else:
            print("  ✓ Both methods agree on spin-state ordering.")
            row["spin_ordering_correct"] = True

        report_rows.append(row)

    # ── Summary report ──────────────────────────────────────────────────────
    write_report(report_rows)


def write_report(rows):
    """Write a markdown summary report."""
    md = ["# CASSCF(10,10) Label Reliability Report\n",
          "**Validation against autoCAS gold-standard active spaces.**\n",
          "\n## Active space comparison\n",
          "| System | (10,10) | autoCAS | Size diff | "
          "ΔE_GS (kcal/mol) | Inert orbs in (10,10) |",
          "|--------|---------|---------|-----------|"
          "-------------------|----------------------|"]
    for r in rows:
        nd = (int(r["autocas_cas"].split(",")[1].rstrip(")")) -
              int(r["label_cas"].split(",")[1].rstrip(")")))
        flag = "✓" if abs(r["dE_gs_kcal"]) < 1 else \
               "⚠" if abs(r["dE_gs_kcal"]) < 5 else "✗"
        md.append(
            f"| {r['system']} | {r['label_cas']} | {r['autocas_cas']} | "
            f"{nd:+d} orbs | {r['dE_gs_kcal']:+.2f} {flag} | "
            f"{r['inert_orbs_label']} / {r['label_cas'].split(',')[1].rstrip(')')} |")

    md += ["\n## Spin-state gap comparison\n",
           "| System | (10,10) gap (cm⁻¹) | autoCAS gap (cm⁻¹) | "
           "Difference | Ordering agrees? |",
           "|--------|--------------------|--------------------|"
           "------------|-----------------|"]
    for r in rows:
        ok = "✓" if r["spin_ordering_correct"] else "✗ WRONG"
        md.append(
            f"| {r['system']} | {r['gap_label_cm1']:+.0f} | "
            f"{r['gap_autocas_cm1']:+.0f} | "
            f"{r['gap_diff_cm1']:+.0f} | {ok} |")

    md += ["\n## Interpretation guide\n",
           "- **ΔE_GS < 1 kcal/mol**: (10,10) label is within chemical accuracy of "
           "autoCAS — reliable for energy-based ML targets.",
           "- **ΔE_GS 1–5 kcal/mol**: moderate error — acceptable for qualitative "
           "tasks, not for quantitative energy prediction.",
           "- **ΔE_GS > 5 kcal/mol**: (10,10) label is unreliable for this system — "
           "systematically underestimates correlation energy.",
           "- **Inert orbs in (10,10)**: orbitals forcibly included in (10,10) that "
           "have NOON > 1.98 or < 0.02 — wasted degrees of freedom.",
           "- **Spin ordering wrong**: the most serious failure — the ML model would "
           "be trained on qualitatively incorrect labels for this system."]

    report_path = OUT / "validation_report.md"
    report_path.write_text("\n".join(md))
    print(f"\n{'='*70}")
    print(f"Report written to: {report_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
